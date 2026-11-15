//! TEMPORARY experiment knobs (env H265K_<NAME>); not for commit.
use std::collections::HashMap;
use std::sync::OnceLock;

static KN: OnceLock<HashMap<String, f64>> = OnceLock::new();

pub(crate) fn k(name: &str, default: f64) -> f64 {
    let m = KN.get_or_init(|| {
        std::env::vars()
            .filter_map(|(k, v)| {
                k.strip_prefix("H265K_")
                    .and_then(|n| v.parse().ok().map(|x| (n.to_string(), x)))
            })
            .collect()
    });
    m.get(name).copied().unwrap_or(default)
}
