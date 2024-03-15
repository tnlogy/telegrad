pub mod nn;

use core::cell::RefMut;
use core::cell::Ref;
use core::cell::RefCell;
use std::rc::Rc;

// to write svg
use std::fs::File;
use std::io::Write;
use graphviz_rust::{
    cmd::Format,
    exec, parse,
    printer::PrinterContext,
};
use graphviz_rust::dot_structures::*;

#[derive(Debug, Clone, Default)]
pub struct NodeData {
    pub value: f32,
    pub grad: f32,
    pub label: String
}

#[derive(Debug, Clone)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow
}

impl Op {
    fn eval(self, left: f32, right: f32) -> f32 {
        match self {
            Op::Add => { left + right }
            Op::Sub => { left - right }
            Op::Mul => { left * right }
            Op::Div => { left / right }
            Op::Pow => { left.powf(right) }
        }
    }

    fn derive(self, stack: &mut Vec<Node>, op: usize, left: usize, right: usize) {
        match self {
            Op::Add | Op::Sub => {
                stack[left].data.grad += stack[op].data.grad;
                stack[right].data.grad += stack[op].data.grad;
            }
            Op::Mul => {
                stack[left].data.grad += stack[right].data.value * stack[op].data.grad;
                stack[right].data.grad += stack[left].data.value * stack[op].data.grad;
            }
            Op::Pow => {
                stack[left].data.grad += (stack[right].data.value * stack[left].data.value.powf(stack[right].data.value - 1.)) * stack[op].data.grad;
            }
            Op::Div => { panic!("Div should be rewritten with NodeStack::div()"); }
        }
    }

    fn to_str<'a>(self) -> &'a str {
        match self {
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mul => "*",
            Op::Pow => "pow",
            Op::Div => "/"
        }
    }
}


#[derive(Debug, Clone)]
pub enum UnaryOp {
    Log,
    ReLu,
    TanH,
    Sigmoid,
    Exp
}

impl UnaryOp {
    fn eval(self, left: f32) -> f32 {
        match self {
            UnaryOp::ReLu => { if left < 0. { 0. } else { left } }
            UnaryOp::TanH => { (f32::exp(2.0 * left) - 1.0) / (f32::exp(2.0 * left) + 1.0) }
            UnaryOp::Sigmoid => { 1.0 / (1.0 + f32::exp(-left)) }
            UnaryOp::Exp => { f32::exp(left) }
            UnaryOp::Log => {
                let epsilon = 1e-6;
                if left.abs() < epsilon {
                    (left + epsilon).ln()
                } else {
                    left.ln()
                }
            }
        }
    }

    fn derive(self, stack: &mut Vec<Node>, op: usize, left: usize) {
        match self {
            UnaryOp::ReLu => {
                stack[left].data.grad += if stack[op].data.value > 0. { stack[op].data.grad } else { 0. };
            }
            UnaryOp::TanH => {
                stack[left].data.grad += (1.0 - stack[op].data.value.powi(2)) * stack[op].data.grad;
            }
            UnaryOp::Sigmoid => {
                stack[left].data.grad += stack[op].data.value * (1.0 - stack[op].data.value) * stack[op].data.grad;
            }
            UnaryOp::Exp => {
                stack[left].data.grad += stack[op].data.value * stack[op].data.grad;
            }
            UnaryOp::Log => {
                let epsilon = 1e-6;
                let d = if stack[op].data.value < epsilon {
                    1.0 / stack[op].data.value + epsilon
                } else {
                    1.0 / stack[op].data.value
                };
                stack[left].data.grad += d * stack[op].data.grad;
            }
        }
    }
    fn to_str<'a>(self) -> &'a str {
        match self {
            UnaryOp::Log => "log",
            UnaryOp::Sigmoid => "sigmoid",
            UnaryOp::Exp => "exp",
            UnaryOp::TanH => "tanh",
            UnaryOp::ReLu => "ReLU"
        }
    }
}

#[derive(Debug, Clone)]
pub enum NodeType {
    Value,
    BinaryOp { op: Op, left: usize, right: usize },
    UnaryOp { op: UnaryOp, left: usize }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub data: NodeData,
    pub node: NodeType
}

#[derive(Debug, Default)]
pub struct NodeStack {
    stack: Vec<Node>,
}

pub type NodeStackPtr = Rc<RefCell<NodeStack>>;

impl NodeStack {
    pub fn new() -> NodeStackPtr {
        Rc::new(RefCell::new(NodeStack::default()))
    }
}

#[derive(Debug, Clone)]
pub struct NodePtr {
    stack: NodeStackPtr,
    i: usize,
}


impl NodePtr {
    pub fn pow(self, v: &NodePtr) -> NodePtr {
        self.stack.create_op(Op::Pow, &self, v)
    }
    pub fn sigmoid(self) -> NodePtr {
        self.stack.create_unary_op(UnaryOp::Sigmoid, &self)
    }
    pub fn ln(self) -> NodePtr {
        self.stack.create_unary_op(UnaryOp::Log, &self)
    }
    pub fn tanh(self) -> NodePtr {
        self.stack.create_unary_op(UnaryOp::TanH, &self)
    }
    pub fn exp(self) -> NodePtr { self.stack.create_unary_op(UnaryOp::Exp, &self) }
}

pub trait NodeOps {
    fn update_params(&self, root: &NodePtr, alpha: f32);
    fn val_labeled(&self, data: f32, label: &str) -> NodePtr;
    fn get_data(&self, node: &NodePtr) -> NodeData;
    fn set_value(&self, node: &NodePtr, value: f32);
    fn create_svg(&self, root: &NodePtr, filename: &str);
    fn backward(&self, root: &NodePtr);
    fn forward(&self, to: &NodePtr);
    fn eval(&self, node: &NodePtr);
    fn zerograd(&self);
    fn get(&self, node: &NodePtr) -> Ref<'_, Node>;
    fn get_mut(&self, node: &NodePtr) -> RefMut<'_, Node>;
    fn val(&self, data: f32) -> NodePtr;
    fn set_label(&self, node: &NodePtr, label: &str);
    fn create_op(&self, op: Op, left: &NodePtr, right: &NodePtr) -> NodePtr;
    fn create_unary_op(&self, op: UnaryOp, left: &NodePtr) -> NodePtr;
}

impl NodeOps for NodeStackPtr {
    fn get(&self, node: &NodePtr) -> Ref<'_, Node> {
        Ref::map(self.borrow(), |this| &this.stack[node.i])
    }

    fn get_data(&self, node: &NodePtr) -> NodeData {
        self.get(&node).data.clone()
    }

    fn get_mut(&self, node: &NodePtr) -> RefMut<'_, Node> {
        RefMut::map(self.borrow_mut(), |this| &mut this.stack[node.i])
    }

    fn val(&self, data: f32) -> NodePtr {
        let node = Node {
            data: NodeData { value: data, ..NodeData::default() },
            node: NodeType::Value
        };
        let i = {
            let mut this = self.borrow_mut();
            this.stack.push(node);
            this.stack.len() - 1
        };

        NodePtr { stack: self.clone(), i }
    }

    fn val_labeled(&self, data: f32, label: &str) -> NodePtr {
        let v = self.val(data);
        self.set_label(&v, label);
        v
    }

    fn set_label(&self, node: &NodePtr, label: &str) {
        self.borrow_mut().stack[node.i].data.label = String::from(label);
    }
    fn set_value(&self, node: &NodePtr, value: f32) {
        self.borrow_mut().stack[node.i].data.value = value;
    }

    fn create_op(&self, op: Op, left: &NodePtr, right: &NodePtr) -> NodePtr {
        if let Op::Div = op { // rewrite as left * right^-1
            let neg1 = self.val(-1.);
            let rh = self.create_op(Op::Pow, right, &neg1);
            return self.create_op(Op::Mul, left, &rh);
        }
        let node = Node {
            data: NodeData::default(),
            node: NodeType::BinaryOp { op, left: left.i, right: right.i }
        };
        let i = {
            let mut this = self.borrow_mut();
            this.stack.push(node);
            this.stack.len() - 1
        };
        let node_ptr = NodePtr { stack: self.clone(), i };
        self.eval(&node_ptr);
        node_ptr
    }

    fn create_unary_op(&self, op: UnaryOp, left: &NodePtr) -> NodePtr {
        let node = Node {
            data: NodeData::default(),
            node: NodeType::UnaryOp { op, left: left.i }
        };
        let i = {
            let mut this = self.borrow_mut();
            this.stack.push(node);
            this.stack.len() - 1
        };
        let node_ptr = NodePtr { stack: self.clone(), i };
        self.eval(&node_ptr);
        node_ptr
    }

    fn zerograd(&self) {
        let mut this = self.borrow_mut();
        for mut n in this.stack.iter_mut() {
            n.data.grad = 0.;
        }
    }

    fn eval(&self, node: &NodePtr) {
        let mut this = self.borrow_mut();
        let n = this.stack[node.i].clone();

        this.stack[node.i].data.value = match n.node {
            NodeType::Value => n.data.value,
            NodeType::BinaryOp { op, left, right} => {
                op.eval(this.stack[left].data.value, this.stack[right].data.value)
            }
            NodeType::UnaryOp { op, left} => {
                op.eval(this.stack[left].data.value)
            }
        }
    }

    fn forward(&self, to: &NodePtr) {
        for i in 0..=to.i {
            self.eval(&NodePtr {i, stack: Rc::clone(&to.stack)});
        }
    }

    fn update_params(&self, root: &NodePtr, alpha: f32) { // Naive update all data params
        let mut this = self.borrow_mut();
        let w_string = String::from("w");
        let b_string = String::from("b");
        //let mut updates = 0;

        for i in 0..=root.i {
            if this.stack[i].data.label == w_string || this.stack[i].data.label == b_string {
                this.stack[i].data.value -= alpha * this.stack[i].data.grad;
                //updates += 1;
            }
        }
        //println!("Updated {} params", updates);
    }

    fn backward(&self, root: &NodePtr) {
        self.zerograd();
        let mut this = self.borrow_mut();
        this.stack[root.i].data.grad = 1.;
        for i in (0..=root.i).rev() {
            let node = this.stack[i].node.clone();
            //println!("{}: {} backward: {:?}", i, &n.data.label, &n);
            match node {
                NodeType::Value => {}
                NodeType::BinaryOp { op, left, right} => {
                    op.derive(&mut this.stack, i, left, right);
                }
                NodeType::UnaryOp { op, left} => {
                    op.derive(&mut this.stack, i, left);
                }
            }
            //println!("{}: {} after backward: {:?}", i, &n.data.label, &n.data);
        }

        /*println!("gradients updated to:");
        for i in (0..=root.i).rev() {
            let n = this.stack[i].clone();
            println!("{} = {:?}", &n.data.label, &n.data.grad);
        }*/
    }

    fn create_svg(&self, root: &NodePtr, filename: &str) {
        let mut this = self.borrow_mut();
        let mut rows: Vec<String> = Vec::new();
        rows.push(String::from(r#"digraph structs {
                rankdir="LR"
                node [shape=record];"#));
        for i in (0..=root.i).rev() {
            let n = this.stack[i].clone();
            rows.push(match n.node {
                NodeType::Value => {
                    format!(r#"node{} [label="{} | value: {:.2} | grad: {:.2}"];"#, i, n.data.label, n.data.value, n.data.grad)
                }
                NodeType::BinaryOp { op, left, right } => {
                    format!(r#"node{} [label="value: {:.2} | grad: {:.2}"];
                node{}_op [label="{}"; shape=circle];
                node{} -> node{}_op; node{}_op -> node{};
                node{}_op -> node{};"#, i, n.data.value, n.data.grad, i, op.to_str(), i, i, i, left, i, right)
                }
                NodeType::UnaryOp { op, left } => {
                    format!(r#"node{} [label="value: {:.2} | grad: {:.2}"];
                node{}_op [label="{}"; shape=square];
                node{} -> node{}_op; node{}_op -> node{};"#, i, n.data.value, n.data.grad, i, op.to_str(), i, i, i, left)
                }
            });
        }
        rows.push(String::from("}"));
        // println!("{}", rows.join(""));
        let g: Graph = parse(&rows.join("")).unwrap();
        let graph_svg = exec(g, &mut PrinterContext::default(), vec![Format::Svg.into()]).unwrap();
        let mut file = File::create(filename).expect("Failed to create file");
        file.write_all(String::from_utf8_lossy(&graph_svg).as_bytes()).expect("Failed to write to file");
        println!("Saved file {}", &filename);
    }
}

#[macro_export]
macro_rules! val {
    ($ns:expr, $var:ident, $value:expr) => {
        let $var = &$ns.val($value);
        $ns.set_label($var, stringify!($var));
    };
}

macro_rules! binary_operator_overload {
    ($op:ident, $f:ident) => {
        impl std::ops::$op<NodePtr> for NodePtr {
            type Output = NodePtr;

            fn $f(self, other: NodePtr) -> NodePtr {
                self.stack.create_op(Op::$op, &self, &other)
            }
        }
        impl std::ops::$op<&NodePtr> for NodePtr {
            type Output = NodePtr;

            fn $f(self, other: &NodePtr) -> NodePtr {
                self.stack.create_op(Op::$op, &self, other)
            }
        }

        impl std::ops::$op<NodePtr> for &NodePtr {
            type Output = NodePtr;

            fn $f(self, other: NodePtr) -> NodePtr {
                self.stack.create_op(Op::$op, self, &other)
            }
        }

        impl std::ops::$op<&NodePtr> for &NodePtr {
            type Output = NodePtr;

            fn $f(self, other: &NodePtr) -> NodePtr {
                self.stack.create_op(Op::$op, self, other)
            }
        }

        impl std::ops::$op<f32> for &NodePtr {
            type Output = NodePtr;

            fn $f(self, other: f32) -> NodePtr {
                let val = self.stack.val(other);
                self.stack.create_op(Op::$op, self, &val)
            }
        }
        impl std::ops::$op<f32> for NodePtr {
            type Output = NodePtr;

            fn $f(self, other: f32) -> NodePtr {
                let val = self.stack.val(other);
                self.stack.create_op(Op::$op, &self, &val)
            }
        }
    }
}
binary_operator_overload!(Add, add);
binary_operator_overload!(Mul, mul);
binary_operator_overload!(Sub, sub);
binary_operator_overload!(Div, div);

#[test]
fn test_mutate_parts() {
    let ns = Rc::new(RefCell::new(NodeStack::default()));
    val!(&ns, w, 1.);
    val!(&ns, b, 1.);
    val!(&ns, x, 1.);

    let f = x * w + b * 3.;
    let g = &f * 10.;
    ns.get_mut(b).data.value = 2.;
    println!("b: {:?}", ns.get(b));
    ns.get_mut(&g).data.value = 2.;
    ns.get_mut(&f).data.value = 2.;
    println!("f = w * x + b: {:?}. g = {:?}", ns.get(&f), ns.get(&g));
}

// Some test similar to the tests in https://github.com/karpathy/nn-zero-to-hero
#[test]
fn test_z_to_h2() {
    let ns = NodeStack::new();
    val!(ns, a, 3.0);
    let b = a + a; ns.set_label(&b, "b");
    ns.backward(&b);
    println!("a {:?}", ns.get_data(a));
    println!("b {:?}", ns.get_data(&b));
    assert!(f32::abs(ns.get_data(a).grad - 2.0) < 0.01);
}

#[test]
fn test_z_to_h3() {
    let ns = NodeStack::new();
    val!(ns, a, -2.0);
    val!(ns, b, 3.0);
    let f = (a * b) * (a + b);
    ns.backward(&f);
    assert!(f32::abs(ns.get_data(a).grad - -3.0) < 0.01);
    assert!(f32::abs(ns.get_data(b).grad - -8.0) < 0.01);
}

#[test]
fn test_z_to_h4() {
    let ns = NodeStack::new();

    val!(ns, x1, 2.0);
    val!(ns, x2, 0.0);
    val!(ns, w1, -3.0);
    val!(ns, w2, 1.0);
    val!(ns, b, 6.8813735870195432);
    let f = x1*w1 + x2*w2 + b;
    let o = &f.tanh();
    ns.set_label(&o, "o");
    ns.backward(&o);

    assert!(f32::abs(ns.get(&x1).data.grad - -1.5) < 0.01);
    assert!(f32::abs(ns.get(&w1).data.grad - 1.0) < 0.01);
    assert!(f32::abs(ns.get(&w2).data.grad - 0.0) < 0.01);
}

#[test]
fn test_z_to_h5() {
    let ns = NodeStack::new();
    val!(ns, x1, 2.0);
    val!(ns, x2, 0.0);
    val!(ns, w1, -3.0);
    val!(ns, w2, 1.0);
    val!(ns, b, 6.8813735870195432);

    let x1w1x2w2 = x1*w1 + x2*w2 + b;
    let n = &x1w1x2w2 * 2.;
    let e = &n.exp(); ns.set_label(&e, "e"); // e = (2*n)^e
    let o = (e - 1.) / (e + 1.);
    ns.backward(&o);
    println!("grad is {:?}", ns.get_data(&x1w1x2w2).grad);
    assert!(f32::abs(ns.get_data(&x1w1x2w2).grad - 0.5) < 0.01);
}

/*
  load data created with
  a = np.array([1, 3.5, -6, 2.3])
  np.save('test-data/plain.npy', a)
*/
pub fn load_npy<T>(filename: &str) -> Vec<Vec<T>>
where T:Clone + npyz::Deserialize {
    let bytes = std::fs::read(filename).expect("Failed to load");
    let npy = npyz::NpyFile::new(&bytes[..]).expect("Failed to parse");
    let shape = npy.shape();
    assert_eq!(shape.len(), 2, "Can only read 2d arrays");
    // assert_eq!(npy.order(), npyz::Order::Fortran);
    let input = npy.clone().into_vec::<T>().expect("data");
    let (rows, cols) = (shape[1] as usize, shape[0] as usize);
    assert_eq!(rows * cols, input.len(), "Shape doesn't match the length of the input vector");

    let mut result = Vec::with_capacity(rows);
    for i in 0..cols {
        let mut sub_vec = Vec::with_capacity(cols);
        for j in 0..rows {
            sub_vec.push(input[j * cols + i].clone());
        }
        result.push(sub_vec);
    }
    result
}
