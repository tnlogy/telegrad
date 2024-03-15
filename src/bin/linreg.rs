use telegrad::*;


fn main() {
    println!("Test simple linear regression");
    fn f_target(x: f32) -> f32 { x * 4. - 5. } // Function to "find" values for w = 4 and b = -5
    let ns = NodeStack::new();

    val!(ns, w, 2.);
    val!(ns, b, 1.);
    val!(ns, x, 1.);
    val!(ns, y_hat, f_target(1.));

    let f = x * w + b;
    let loss = (&f - y_hat).pow(&ns.val(2.)); // squared error term

    println!("f = w * x + b: {:?}. loss: {:?}", ns.get(&f).data, ns.get(&loss).data);

    let alpha: f32 = 0.01;

    for epoch in 0..200 {
        let mut total_loss = 0.;
        for x_in  in (0..=10).map(|x| x as f32) {
            ns.set_value(&x, x_in);
            ns.set_value(&y_hat, f_target(x_in));

            ns.forward(&loss);
            ns.backward(&loss);
            total_loss += ns.get(&loss).data.value;

            ns.get_mut(&w).data.value -= alpha * ns.get_data(&w).grad;
            ns.get_mut(&b).data.value -= alpha * ns.get_data(&b).grad;
        }
        if epoch % 20 == 0 {
            println!("epoch: {} total_loss: {}", epoch, total_loss);
        }
    }
    println!("model values of w = {:.2} and b = {:.2}", ns.get(&w).data.value, ns.get(&b).data.value);
    println!("actual values of w = 4.0 and b = -5.0");
    assert!(f32::abs(ns.get(&w).data.value - 4.0) < 0.01);
    assert!(f32::abs(ns.get(&b).data.value - -5.0) < 0.01);
    ns.create_svg(&loss, "linear_regression.svg");
}
