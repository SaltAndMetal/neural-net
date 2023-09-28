#[allow(non_snake_case, dead_code)]
extern crate rand;
use rand::Rng;

use std::fs;
use std::fs::File;
use std::io::prelude::*;

mod neuron;

use crate::network::neuron::Neuron;

use std::sync::{Arc, Mutex};

extern crate crossbeam;
use crossbeam_utils::thread;

const STEP_SIZE: f64 = 1.0;

#[derive(Debug)]
struct Adjustment {
    weights: Vec<f64>,
    bias: f64
}

#[derive(Debug)]
pub struct Network {
    //Right to left
    neurons: Vec<Vec<Neuron>>,
}

impl Network {
    //First element of sizes is the size of the input vector, and does not create a proper layer.
    pub fn new(sizes: &Vec<u32>) -> Network {
        let mut rng = rand::thread_rng();
        let mut neurons = Vec::new();
        let sizes_iter = sizes.windows(2);
        for i in sizes_iter {
            let mut layer = Vec::new();
            for _ in 0..i[0] {
                let mut weights = Vec::new();
                for _ in 0..i[1] {
                    weights.push(rng.gen_range(-1.0..1.0));
                }
                layer.push(Neuron::new(weights, rng.gen_range(-1.0..1.0)));
            }
            neurons.push(layer);
        }
        Network { neurons }
    }

    pub fn customNetwork(neurons: Vec<Vec<(Vec<f64>, f64)>>) -> Network {
        let mut neurons_ = Vec::new();
        for layer in neurons {
            let mut neurons__ = Vec::new();
            for neuron in layer {
                neurons__.push(Neuron::new( neuron.0, neuron.1));
            }
            neurons_.push(neurons__);
        }
        Network{ neurons: neurons_ }
    }

    pub fn fromFile(filename: &str) -> Network {
        let mut network = fs::read(filename).expect("Could not read file");
        let layerNum = u32::from_le_bytes(network.drain(0..4).collect::<Vec<u8>>()[..].try_into().unwrap());
        let mut sizes = Vec::new();
        for _ in 0..layerNum {
            sizes.push(u32::from_le_bytes(network.drain(0..4).collect::<Vec<u8>>()[..].try_into().unwrap()));
        }
        let mut neurons = Vec::new();
        for i in 0..(sizes.len()-1) {
            neurons.push(Vec::new());
            for neuron in 0..sizes[i] {
                let mut weights = Vec::new();
                for _ in 0..sizes[i+1] {
                    weights.push(f64::from_le_bytes(network.drain(0..8).collect::<Vec<u8>>()[..].try_into().unwrap()));
                }
                neurons[i].push(Neuron::new(weights, f64::from_le_bytes(network.drain(0..8).collect::<Vec<u8>>()[..].try_into().unwrap())));
            }
        }
        Network{ neurons }

    }

    pub fn write(&self) {
        let mut file = File::create("network").expect("Could not create file");
        let mut value: Vec<u8> = Vec::new();
        value.append(&mut (((self.neurons.len()+1) as u32).to_le_bytes().to_vec()));
        for layer in &self.neurons {
            value.append(&mut ((layer.len() as u32).to_le_bytes().to_vec()));
        }
        value.append(&mut ((self.neurons[self.neurons.len()-1][0].weights.len() as u32).to_le_bytes().to_vec()));
        for layer in &self.neurons {
            for neuron in layer {
                for weight in &neuron.weights {
                    value.append(&mut weight.to_le_bytes().to_vec());
                }
                value.append(&mut neuron.bias.to_le_bytes().to_vec());
            }
        }
        file.write_all(&value);
    }

    //The main work of the network. Computes answers.
    pub fn process(&mut self, initialInputs: &Vec<f64>) -> Vec<f64> {
        let layers = self.neurons.len();
        for neuron in self.neurons[layers - 1].iter_mut() {
            neuron.setInputs(initialInputs.to_vec());
            neuron.process();
        }
        let mut i: usize = 1;
        let len = self.neurons.len();
        let mut prev = self.neurons[len - i].clone();
        while i < len {
            for neuron in &mut self.neurons[len - i - 1] {
                neuron.readInputs(&prev);
                neuron.process();
            }
            i += 1;
            prev = self.neurons[len - i].clone();
        }
        let mut output = Vec::new();
        for neuron in &self.neurons[0] {
            output.push(neuron.getOutput());
        }
        output
    }

   pub fn cost(computed: &Vec<f64>, desired: &Vec<f64>) -> f64 {
        let mut runningTotal = 0.0;
        for (x, y) in computed.iter().zip(desired.iter()) {
            runningTotal += (x-y).powf(2.0_f64);
        }
        runningTotal
    }

   //Panics if desiredOuputs is shorter than inputs. If inputs is shorter than desiredOutputs, it
   //truncates it. Returns the average cost.
    pub fn batch(&mut self, step_factor: f64, inputs: &Vec<Vec<f64>>, desiredOutputs: &Vec<Vec<f64>>) -> f64 {
        let wrappedSelf = Arc::new(Mutex::new(&mut *self));
        let mut adjustments = Vec::new();
        let mut cost = 0.0;
        thread::scope(|s| {
            let mut handles = Vec::new();
            for i in 0..inputs.len() {
                let wrappedSelfCloned = wrappedSelf.clone();
                handles.push(s.spawn(move |_| {
                    let output = wrappedSelfCloned.lock().unwrap().process(&inputs[i]);
                    let newCost = Self::cost(&output, &desiredOutputs[i]);
                    (newCost, wrappedSelfCloned.lock().unwrap().backpropogate(STEP_SIZE/inputs.len() as f64, &desiredOutputs[i]))
                }));
            }
            for handle in handles {
                let outputs = handle.join().unwrap();
                adjustments.push(outputs.1);
                cost += outputs.0;
            }
        }).unwrap();
        
        for adjustment in &adjustments {
            self.adjust(&adjustment);
        }

        cost/inputs.len() as f64

    }

    fn backpropogate(&mut self, step_size: f64, desired: &Vec<f64>) -> Vec<Vec<Adjustment>> {
        let mut importances = Vec::new();
        for (neuron, desire) in self.neurons[0].iter().zip(desired.iter()) {
            importances.push(2.0*(neuron.getOutput()-desire));
        }
        let iter = self.neurons.iter();
        let mut iter_next = self.neurons.iter();
        iter_next.next();
        let mut adjustments = Vec::new();
        for (layer, prev_layer) in iter.zip(iter_next) {
            let result = Self::backpropogate_single_layer(step_size, importances, prev_layer, layer);
            importances = result.0;
            adjustments.push(result.1);
        }
        adjustments
    }

    fn adjust(&mut self, adjustment: &Vec<Vec<Adjustment>>) {
        for (layer, adjustmentLayer) in self.neurons.iter_mut().zip(adjustment.iter()) {
            for (neuron, adjustment) in layer.iter_mut().zip(adjustmentLayer.iter()) {
                neuron.bias -= adjustment.bias;
                for (weight, adjustmentWeight) in neuron.weights.iter_mut().zip(adjustment.weights.iter()) {
                    *weight -= adjustmentWeight;
                }
            }
        }
    }

    //Must be called after process. Reads the last outputs from neurons, return the adjustments to
    //be made to the layer and the sensitivities for all outputs from the layer before.
    fn backpropogate_single_layer(step_size: f64, importances: Vec<f64>, prev_layer: &Vec<Neuron>, layer: &Vec<Neuron>) -> (Vec<f64>, Vec<Adjustment>) {
        let mut adjustments = Vec::new();
        let mut nextImportances = Vec::new();
        for (neuron, importance) in layer.iter().zip(importances.iter()) {
            let coefficiant = step_size * importance * Neuron::dSigmoid(&neuron.preSigmoidOutput);
            let mut weights = Vec::new();
            for previous in prev_layer {
                weights.push(previous.getOutput() * coefficiant);
            }
            adjustments.push(Adjustment{ weights, bias: coefficiant });
        }
        for i in 0..prev_layer.len() {
            let mut importance = 0.0;
            for (j, neuron) in layer.iter().enumerate() {
                importance += neuron.weights[i] * importances[j];
            }
            nextImportances.push(importance);
        }
        (nextImportances, adjustments)
    }
}
