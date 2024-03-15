use rand::Rng;
use crate::*;

// Neural network
#[derive(Debug)]
pub struct Neuron {
    pub input: Vec<NodePtr>,
    pub weights: Vec<NodePtr>,
    pub bias: NodePtr,
    pub output: NodePtr
}

impl Neuron {
    pub fn new(ns: NodeStackPtr, input: Vec<NodePtr>, activation: UnaryOp) -> Self {
        let mut rng = rand::thread_rng();
        // let mut input: Vec<NodePtr> = (0..number_of_inputs).map(|_| ns.val_labeled(1.0, "x")).collect();
        let number_of_inputs = input.len();
        let mut weights: Vec<NodePtr> = (0..number_of_inputs).map(|_| ns.val_labeled( rng.gen_range(-1.0..=1.0), "w")).collect();
        let mut output = &input[0] * &weights[0];
        for (x, w) in input.iter().skip(1).zip(weights.iter().skip(1)) {
            output = output + x * w;
        }
        let bias = ns.val_labeled(0.0, "b");
        output = output + &bias;
        output = ns.create_unary_op(activation, &output);
        Neuron { input, weights, bias, output }
    }
}

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub outputs: Vec<NodePtr>
}

impl Layer {
    pub fn new(ns: NodeStackPtr, input: Vec<NodePtr>, number_of_outputs: usize, activation: UnaryOp) -> Self {
        let neurons: Vec<Neuron> = (0..number_of_outputs).map(|_| Neuron::new(ns.clone(), input.clone(), activation.clone())).collect();  
        let outputs = neurons.iter().map(|n| n.output.clone()).collect();
        Layer { neurons, outputs }
    }
}
