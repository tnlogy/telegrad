use telegrad::*;

fn main() {
    println!("Test simple logistic regression");
    // example with 6 training examples with 2 features
    let x_train: Vec<Vec<f32>> = vec![
        vec![0.5, 1.5],
        vec![1.0, 1.0],
        vec![1.5, 0.5],
        vec![3.0, 0.5],
        vec![2.0, 2.0],
        vec![1.0, 2.5],
    ];
    let y_train: Vec<f32> = vec![0., 0., 0., 1., 1., 1.];
    let ns = NodeStack::new();

    // first feature
    val!(ns, w1, 1.0); // 5.28);
    val!(ns, x1, x_train[0][0]);
    // second feature
    val!(ns, w2, 1.0); // 5.08);
    val!(ns, x2, x_train[0][1]);
    val!(ns, b, 1.0); // -14.22); // bias
    let f_lin = x1 * w1 + x2 * w2 + b;
    let f = f_lin.sigmoid();
    // val!(ns, f, 1.);
    // loss = -y_hat*log(f) - (1-y_hat)*np.log(1-f)
    val!(ns, y_hat, y_train[0]);
    let loss = (&f - y_hat).pow(&ns.val(2.)); ns.set_label(&loss, "loss");
    // TODO: bug - something strange with f.ln(ns) or its numerical derivative
    // val!(ns, one, 1.0); let loss = y_hat * f.ln(ns) - (one - y_hat) * (one - f).ln(ns);
    println!("f = {:?} real_f = {:?}", ns.get_data(&f).value, y_train[0]);
    let out = ns.get_data(&f).value;
    let real_loss = -y_train[0] * out.ln() - (1.0 - y_train[0]) * (1.0 - out).ln();
    println!("out: {} loss: {:?} ({:?})", out.ln(), ns.get_data(&loss).value, real_loss);

    let alpha: f32 = 0.01;
    for epoch in 0..10000 {
        let mut total_loss = 0.;
        for (x, y) in x_train.iter().zip(y_train.iter()) {
            ns.get_mut(x1).data.value = x[0];
            ns.get_mut(x2).data.value = x[1];
            ns.get_mut(y_hat).data.value = *y;

            ns.forward(&loss);
            ns.backward(&loss);
            total_loss += ns.get_data(&loss).value;

            // update parameters
            ns.get_mut(&w1).data.value -= alpha * ns.get_data(w1).grad;
            ns.get_mut(&w2).data.value -= alpha * ns.get_data(w2).grad;
            ns.get_mut(&b).data.value -= alpha * ns.get_data(b).grad;
        }
        if epoch % 20 == 0 {
            println!("epoch: {} total_loss: {}", epoch, total_loss);
        }
    }
    println!("model values of w1, w2 = {:.2}, {:.2} and b = {:.2}", ns.get(&w1).data.value, ns.get(&w2).data.value, ns.get(&b).data.value);
    println!("actual parameters: w1, w2 = 5.28, 5.08, b = -14.222409982019837");
    for (x, y) in x_train.iter().zip(y_train.iter()) {
        ns.get_mut(x1).data.value = x[0];
        ns.get_mut(x2).data.value = x[1];
        ns.forward(&f);
        println!("f = {:?} actual = {:?}", if ns.get_data(&f).value > 0.5 { 1.0 } else { 0.0 }, y);
    }
    ns.create_svg(&loss, "logistic_regression.svg");
}

