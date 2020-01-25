use std::collections::{HashSet, HashMap};
use rand::prelude::*;
use rayon::prelude::*;


type Tags = HashSet<usize>;

#[derive(Debug)]
struct Input {
    vertical_photos: Vec<Tags>,
    horisontal_photos: Vec<Tags>,
    photos: Vec<Tags>,
    v_ids: Vec<usize>,
    h_ids: Vec<usize>,
    tag_count: usize,
}

impl Input {
    fn parse(s: &str) -> Input {
        let mut tag_map = HashMap::new();
        let mut tag_i   = 0;
        let mut id = 0;

        let mut input = Input {
            vertical_photos: Vec::new(),
            horisontal_photos: Vec::new(),
            photos: Vec::new(),
            v_ids: Vec::new(),
            h_ids: Vec::new(),
            tag_count: 0,
        };

        for line in s.lines().skip(1) {
            let mut ws = line.split_whitespace();
            let ori = ws.next().unwrap();
            ws.next();

            let tags = ws.map(|t| {
                *tag_map.entry(t).or_insert_with(|| {
                    tag_i += 1;
                    tag_i
                })
            }).collect::<Tags>();

            input.photos.push(tags.clone());
            if ori == "H" {
                input.horisontal_photos.push(tags);
                input.h_ids.push(id);
            }
            else {
                input.vertical_photos.push(tags);
                input.v_ids.push(id);
            }

            id += 1;
        }

        input.tag_count = tag_i;

        input
    }
}

struct Output {
    photos: Vec<Vec<usize>>,
    score: u64,
}

impl Output {
    fn print(&self) -> (u64, String) {
        let mut res = String::new();
        res += &format!("{}\n", self.photos.len());

        for ps in &self.photos {
            if ps.len() == 1 {
                res += &format!("{}\n", ps[0]);
            }
            else if ps.len() == 2 {
                res += &format!("{} {}\n", ps[0], ps[1]);
            }
        }

        (self.score, res)
    }

    fn compute_score(&mut self, input: &Input) -> u64 {
        let mut score = 0;
        for i in 0..self.photos.len()-1 {
            let tags_1 = self.photos[i].iter()
                .flat_map(|i| &input.photos[*i])
                .map(|x| *x)
                .collect::<Tags>();
            let tags_2 = self.photos[i+1].iter()
                .flat_map(|i| &input.photos[*i])
                .map(|x| *x)
                .collect::<Tags>();

            score += transition_score(&tags_1, &tags_2);
        }

        self.score = score;
        score
    }
}

struct State {
    input: Input,
    used: Vec<bool>,
    h_photos: HashMap<usize, Tags>,
    v_photos: HashMap<usize, Tags>,
    available_h: HashSet<usize>,
    available_v: HashSet<usize>,
    tag_to_image_h: HashMap<usize, Vec<usize>>,
    tag_to_image_v: HashMap<usize, Vec<usize>>,
}

impl State {
    fn use_photo(&mut self, photo: usize) {
        self.used[photo] = true;
        self.available_v.remove(&photo);
        self.available_h.remove(&photo);
    }
}

fn transition_score(a: &Tags, b: &Tags) -> u64 {
    let union_count = a.intersection(b).count() as u64;
    std::cmp::min(union_count, std::cmp::min(a.len() as u64 - union_count, b.len() as u64 - union_count))
}

fn select_next(tags: &Tags, state: &State) -> (u64, Vec<usize>) {
    let mut rng = rand::thread_rng();
    let mut h_buf = vec![0; 100000];
    let mut v_buf = vec![0; 1000];

    let h_count = state.available_h.iter().map(|x| *x).choose_multiple_fill(&mut rng, &mut h_buf);
    let v_count = state.available_v.iter().map(|x| *x).choose_multiple_fill(&mut rng, &mut v_buf);

    let mut best = Vec::new();
    let mut best_score = 0; //transition_score(&tags, state.h_photos.get(&h_buf[0]).unwrap());

    match (0..h_count).into_par_iter().map(|i| {
        let score = transition_score(&tags, state.h_photos.get(&h_buf[i]).unwrap());
        (score, vec![h_buf[i]])
    }).max() {
        Some((score, x)) => if score >= best_score {
            best = x;
            best_score = score;
        }
        None => ()
    }

    /*
    for i in 0..h_count {
        //let score = transition_score(&tags, state.input.horisontal_photos.get(h_buf[i]).unwrap());
        let score = transition_score(&tags, state.h_photos.get(&h_buf[i]).unwrap());
        if score >= best_score {
            best = vec![h_buf[i]];
            best_score = score;
        }
    }*/
    match (0..v_count).into_par_iter().flat_map(|i| (i+1..v_count).into_par_iter().map(move |j| (i, j))).map(|(i, j)| {
        let tags2 = state.v_photos.get(&v_buf[i]).unwrap()
            .union(&state.v_photos.get(&v_buf[j]).unwrap()).map(|x| *x).collect::<Tags>();

        let score = transition_score(&tags, &tags2);
        (score, vec![v_buf[i], v_buf[j]])
    }).max() {
        Some((score, x)) => if score >= best_score {
            best = x;
            best_score = score;
        }
        None => ()
    }
    /*
    for i in 0..v_count {
        for j in i+1..v_count {
            let tags2 = state.v_photos.get(&v_buf[i]).unwrap()
                .union(&state.v_photos.get(&v_buf[j]).unwrap()).map(|x| *x).collect::<Tags>();

            let score = transition_score(&tags, &tags2);
            if score >= best_score {
                best = vec![v_buf[i], v_buf[j]];
                best_score = score;
            }
        }
    }*/

    (best_score, best)
}


fn select_next2(tags: &Tags, state: &State) -> (u64, Vec<usize>) {
    let mut h_pot = HashSet::new();
    let mut v_pot = HashSet::new();

    let v = Vec::new();
    for tag in tags {
        for id in state.tag_to_image_h.get(tag).unwrap_or(&v) {
            if state.available_h.contains(id) {
                h_pot.insert(id);
            }
        }
        for id in state.tag_to_image_v.get(tag).unwrap_or(&v) {
            if state.available_v.contains(id) {
                v_pot.insert(id);
            }
        }
    }

    let mut best = Vec::new();
    let mut best_score = 0; //transition_score(&tags, state.h_photos.get(&h_buf[0]).unwrap());

    match h_pot.into_par_iter().map(|i| {
        let score = transition_score(&tags, state.h_photos.get(&i).unwrap());
        (score, vec![*i])
    }).max() {
        Some((score, x)) => if score >= best_score {
            best = x;
            best_score = score;
        }
        None => ()
    }

    match v_pot.par_iter().flat_map(|i| v_pot.par_iter().map(move |j| (i, j))).filter(|(i, j)| i != j).map(|(i, j)| {
        let tags2 = state.v_photos.get(i).unwrap()
            .union(&state.v_photos.get(j).unwrap()).map(|x| *x).collect::<Tags>();

        let score = transition_score(&tags, &tags2);
        (score, vec![**i, **j])
    }).max() {
        Some((score, x)) => if score >= best_score {
            best = x;
            best_score = score;
        }
        None => ()
    }

    if best.len() == 0 {
        return select_next(tags, state)
    }

    (best_score, best)
}

fn greedy_01(input: Input) -> Output {
    let used = vec![false; input.horisontal_photos.len() + input.vertical_photos.len()];
    let available_h = input.h_ids.iter().map(|x| *x).collect::<HashSet<_>>();
    let available_v = input.v_ids.iter().map(|x| *x).collect::<HashSet<_>>();

    let h_photos = input.h_ids.iter().map(|x| *x).zip(
        input.horisontal_photos.iter().map(|x| x.clone()))
        .collect::<HashMap<_, _>>();

    let v_photos = input.v_ids.iter().map(|x| *x).zip(
        input.vertical_photos.iter().map(|x| x.clone()))
        .collect::<HashMap<_, _>>();

    let mut tag_to_image_h = HashMap::new();
    for id in &input.h_ids {
        for tag in h_photos.get(id).unwrap() {
            tag_to_image_h.entry(*tag).or_insert(Vec::new()).push(*id);
        }
    }
    let mut tag_to_image_v = HashMap::new();
    for id in &input.v_ids {
        for tag in v_photos.get(id).unwrap() {
            tag_to_image_v.entry(*tag).or_insert(Vec::new()).push(*id);
        }
    }

    let mut state = State {
        input, used, available_h, available_v, h_photos, v_photos, tag_to_image_h, tag_to_image_v
    };

    let mut photos = Vec::new();

    let mut curr = if state.input.h_ids.len() == 0 {
        state.use_photo(state.input.v_ids[0]);
        state.use_photo(state.input.v_ids[1]);
        photos.push(vec![state.input.v_ids[0], state.input.v_ids[1]]);
        state.input.vertical_photos[0].union(&state.input.vertical_photos[1]).map(|x| *x).collect::<Tags>()
    }
    else {
        state.use_photo(state.input.h_ids[0]);
        photos.push(vec![state.input.h_ids[0]]);
        state.input.horisontal_photos[0].clone()
    };
    let mut score = 0;

    while state.available_h.len() > 0 || state.available_v.len() > 1 {
        /// select_next here means optimizing for few tags per photo
        /// select_next2 here means optimizing for few tags per photo
        let (sc, next) = select_next2(&curr, &state);

        if next.len() == 0 {
            eprintln!("INTERNAL ERROR");
            break;
        }

        score += sc;
        if next.len() == 1 {
            state.use_photo(next[0]);
            curr = state.h_photos.get(&next[0]).unwrap().clone();
            photos.push(next);
        }
        else {
            state.use_photo(next[0]);
            state.use_photo(next[1]);
            curr = state.v_photos.get(&next[0]).unwrap()
                .union(state.v_photos.get(&next[1]).unwrap())
                .map(|x| *x)
                .collect::<Tags>();
            photos.push(next);
        }
    }

    Output {
        photos,
        score
    }
}

fn one_random_01(input: &Input) -> Output {
    let mut rng = rand::thread_rng();
    let mut v_ids = input.v_ids.clone();
    v_ids.shuffle(&mut rng);

    let mut ids = Vec::new();
    for i in (0..v_ids.len()).step_by(1) {
        if i+1 < v_ids.len() {
            ids.push(vec![v_ids[i], v_ids[i+1]]);
        }
    }

    for i in &input.h_ids {
        ids.push(vec![*i]);
    }

    ids.shuffle(&mut rng);

    let mut output = Output {
        photos: ids,
        score: 0,
    };

    output.compute_score(&input);
    output
}

fn many_random_01(input: &Input) -> Output {
    let best = std::sync::Mutex::new(one_random_01(input));
    rayon::scope(|scope| {
        for _ in 0..32 {
            scope.spawn(|_| {
                for _ in 0..100 {
                    let output = one_random_01(input);

                    let mut best_out = best.lock().unwrap();
                    if output.score > best_out.score {
                        *best_out = output;
                    }
                }
            });
        }
    });

    best.into_inner().unwrap()
}

/// This is the callable public function for generating a solution file from an input file
pub fn solution(input: &str, strength: f64) -> (u64, String) {
    let input = Input::parse(input);

    //eprintln!("Input: {:?}", input);

    greedy_01(input).print()
    //many_random_01(&input).print()
}
