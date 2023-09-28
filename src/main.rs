#[allow(non_snake_case, dead_code)]

extern crate rand;
use rand::Rng;

use std::fs;
use std::fs::File;

mod network;
use crate::network::Network;

use std::time::Duration;
use std::thread::sleep;

#[derive(PartialEq)]
enum Mode {
    Test,
    Train
}

fn main() {
    let mode: Mode = Mode::Train;
    let mut network;
    if mode == Mode::Train {
        let mut rng = rand::thread_rng();
        let sizes: Vec<u32> = vec![10, 500, 500, 784];
        network = Network::new(&sizes);
        let mut labels = fs::read("train-lables.idx1-ubyte").expect("Could not read file");
        let mut images = fs::read("train-images.idx3-ubyte").expect("Could not read file");
        labels.drain(0..8);
        images.drain(0..16);
        let mut cost: f64 = 10.0;
        let max_cost: f64  = 10.0;
        let mut prev_cost = max_cost;
        loop {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for _ in 0..500 {
                if images.len() < 784 {
                    images = fs::read("train-images.idx3-ubyte").expect("Could not read file");
                    labels = fs::read("train-lables.idx1-ubyte").expect("Could not read file");
                    labels.drain(0..8);
                    images.drain(0..16);
                }
                let input: Vec<f64> = images.drain(0..784).map(|x| x as f64 / 256.0).collect();
                let mut output = vec![0.0; 10];
                output[labels.remove(0) as usize] = 1.0;
                inputs.push(input);
                outputs.push(output);
            }
            cost = network.batch((cost/max_cost).powf(2.0_f64), &inputs, &outputs);
            println!("{:.3}", cost);
            if cost < 0.15 {
                break;
            }
        }
        network.write();
    }
    else {
        network = Network::fromFile("network");
    }
    let mut labels = fs::read("t10k-labels.idx1-ubyte").expect("Could not read file");
    let mut images = fs::read("t10k-images.idx3-ubyte").expect("Could not read file");
    labels.drain(0..8);
    images.drain(0..16);
    let mut counter = [0; 2];
    for _ in 0..10000 {
        let answers: Vec<f64> = network.process(&images.drain(0..784).map(|x| x as f64 / 256.0).collect());
        let desiredAnswer = labels.remove(0);
        let mut highest: (f64, usize) = (0.0, 0);
        for i in 0..answers.len() {
            if answers[i] > highest.0 {
                highest.0 = answers[i];
                highest.1 = i;
            }
        }
        counter[1] += 1;
        if highest.1 == desiredAnswer as usize{
            counter[0] += 1;
        }
    }
    println!{"{} out of {}", counter[0], counter[1]};
}

