mod problem;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let input = std::fs::read_to_string(&args[1]).unwrap();

    let (score, output) = problem::solution(&input, 1.0);

    eprintln!("Found solution with score: {}", score);

    std::fs::write(&args[2], output).unwrap();

    /*
    let mut best_points = 0;
    let mut strength = 0.09;

    let input = std::fs::read_to_string(&args[1]).unwrap();

    while strength < 1.0 {
        eprint!("\rTrying strength: {:.2}, best solution has: {} points", strength, best_points);
        let (points, ans) = problem::solution(&input, strength);
        if points > best_points {
            std::fs::write(&args[2], ans).unwrap();
            best_points = points;
        }
        
        strength += 0.1;
    }
    eprintln!("\rFound solution with {} points", best_points);*/
}
