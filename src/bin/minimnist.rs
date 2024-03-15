use telegrad::{nn::{Layer, Neuron}, *};

fn main() {
    // Example of training a network to recognize 0/1 for handwritten images of numbers. 
    // Load small mnist dataset
    let x_train: Vec<Vec<f64>> = load_npy("data/X.npy");
    let y_train: Vec<Vec<u8>> = load_npy("data/y.npy");
    let m = 1000; // training examples, just 0 and 1
    // println!("y: {:?} y: {:?}", &y_train[0], &x_train[0]);

    let ns = NodeStack::new();
    let input: Vec<NodePtr> = x_train[0].iter().map(|v| ns.val_labeled(*v as f32, "x")).collect();
    let l1 = Layer::new(ns.clone(), input.clone(), 25, UnaryOp::ReLu);
    let l2 = Layer::new(ns.clone(), l1.outputs, 15, UnaryOp::ReLu);
    let l3 = Layer::new(ns.clone(), l2.outputs, 10, UnaryOp::ReLu);
    let l4 = Layer::new(ns.clone(), l3.outputs, 1, UnaryOp::Sigmoid);

    val!(ns, y_hat, y_train[0][0] as f32);
    let loss = (&l4.outputs[0] - y_hat).pow(&ns.val(2.));
    ns.set_label(&loss, "loss");

    let alpha: f32 = 0.01;
    for epoch in 0..50 {
        let mut total_loss = 0.;
        for j in 0..m { // m training examples
            for i in 0..x_train[j].len() {
                ns.get_mut(&input[i]).data.value = x_train[j][i] as f32;
            }
    
            ns.get_mut(&y_hat).data.value = y_train[j][0] as f32; 
            // ns.forward(&l3.outputs[0]);
            //println!("y_train {} out: {}", &y_train[j], ns.get_data(&l3.outputs[0]).value);
            ns.forward(&loss);
            ns.backward(&loss);
            total_loss += ns.get_data(&loss).value;

            // update parameters
            ns.update_params(&loss, alpha);
            // ns.get_mut(&w1).data.value -= alpha * ns.get_data(w1).grad;
        }
        println!("epoch: {} total_loss: {}", epoch, total_loss);
    }

    println!("After training:");
    for j in 0..10 { // 4 training examples
        for i in 0..x_train[j].len() {
            ns.get_mut(&input[i]).data.value = x_train[j][i] as f32;
        }
        ns.forward(&l4.outputs[0]);
        println!("y_train {} out: {}", &y_train[j][0], ns.get_data(&l4.outputs[0]).value);
    }
    for j in m-10..m { // 4 training examples
        for i in 0..x_train[j].len() {
            ns.get_mut(&input[i]).data.value = x_train[j][i] as f32;
        }
        ns.forward(&l4.outputs[0]);
        println!("y_train {} out: {}", &y_train[j][0], ns.get_data(&l4.outputs[0]).value);
    }

    ns.create_svg(&l4.outputs[0], "mini_mnist_model.svg");
}



#[test]
fn test_neuron() {
    let ns = NodeStack::new();
    let input = (0..2).map(|_| ns.val_labeled(1., "x")).collect();
    let n = Neuron::new(ns.clone(), input, UnaryOp::ReLu);
    ns.create_svg(&n.output, "neuron.svg");
    println!("neuron: {:?}", &n);
}

#[test]
fn test_layer() {
    let ns = NodeStack::new();
    let input = (0..4).map(|v| ns.val_labeled(v as f32, "x")).collect();
    let l = Layer::new(ns.clone(), input, 10, UnaryOp::ReLu);
    
    let mut sum = ns.val(0.);
    for o in l.outputs.iter() {
        println!("sum");
        sum = sum + o;
    }

   
    ns.backward(&sum);
    ns.create_svg(&sum, "layer.svg");
    println!("layer: {:?}", ns.get_data(&sum));
    println!("w_1: {:?}", ns.get_data(&l.neurons[0].weights[0]));
}


#[test]
fn test_sequential_model() {
    // Train a simple sequential NN model
    let ns = NodeStack::new();
    let input: Vec<NodePtr> = (0..2).map(|v| ns.val_labeled(v as f32, "x")).collect();
    let l1 = Layer::new(ns.clone(), input.clone(), 4, UnaryOp::TanH);
    let l2 = Layer::new(ns.clone(), l1.outputs, 4, UnaryOp::TanH);
    let l3 = Layer::new(ns.clone(), l2.outputs, 1, UnaryOp::TanH);

    
    let x_train: [[f32; 3]; 4] = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];

    let y_train: [f32; 4] = [1.0, -1.0, -1.0, 1.0];

    val!(ns, y_hat, y_train[0]);
    let loss = (&l3.outputs[0] - y_hat).pow(&ns.val(2.)); ns.set_label(&loss, "loss");

    let alpha: f32 = 0.01;
    for epoch in 0..1000 {
        let mut total_loss = 0.;
        for j in 0..3 { // 4 training examples
            for i in 0..2 {
                ns.get_mut(&input[i]).data.value = x_train[j][i];
            }
    
            ns.get_mut(&y_hat).data.value = y_train[j]; 
            // ns.forward(&l3.outputs[0]);
            //println!("y_train {} out: {}", &y_train[j], ns.get_data(&l3.outputs[0]).value);
            ns.forward(&loss);
            ns.backward(&loss);
            total_loss += ns.get_data(&loss).value;

            // update parameters
            ns.update_params(&loss, alpha);
            // ns.get_mut(&w1).data.value -= alpha * ns.get_data(w1).grad;
        }
        if epoch % 20 == 0 {
            println!("epoch: {} total_loss: {}", epoch, total_loss);
        }
    }

    println!("After training:");
    for j in 0..3 { // 4 training examples
        for i in 0..2 {
            ns.get_mut(&input[i]).data.value = x_train[j][i];
        }
        ns.forward(&l3.outputs[0]);
        println!("y_train {} out: {}", &y_train[j], ns.get_data(&l3.outputs[0]).value);
    }

    ns.create_svg(&l3.outputs[0], "sequential_model.svg");
}
