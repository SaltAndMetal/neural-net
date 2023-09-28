use std::f64::consts::E;

#[derive(Clone, Debug)]
pub struct Neuron {
    pub bias: f64,
    pub weights: Vec<f64>,
    inputs: Vec<f64>,
    output: f64,
    pub preSigmoidOutput: f64,
}

impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64) -> Neuron {
        Neuron {
            weights,
            bias,
            inputs: Vec::new(),
            output: 0.0,
            preSigmoidOutput: 0.0
        }
    }

    fn sigmoid(x: &f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    pub fn dSigmoid(x: &f64) -> f64 {
        E.powf(-x)/2.0_f64.powf(E.powf(-x) + 1.0)
    }

    //Input vector must be empty when calling
    pub fn readInputs(&mut self, predecessors: &Vec<Neuron>) {
        for neuron in predecessors {
            self.inputs.push(neuron.output)
        }
    }

    pub fn setInputs(&mut self, inputs: Vec<f64>) {
        self.inputs = inputs;
    }

    pub fn getOutput(&self) -> f64 {
        self.output
    }

    pub fn process(&mut self) {
        let mut runningTotal = self.bias;
        for (input, weight) in self.inputs.iter().zip(self.weights.iter()) {
            runningTotal += input * weight;
        }
        self.preSigmoidOutput = runningTotal;
        self.output = Self::sigmoid(&runningTotal);
        self.inputs = Vec::new();
    }
}
