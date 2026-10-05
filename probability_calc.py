import math
import random
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Probability Lab", page_icon="🪐")

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 20% 0%,#2A1065 0%,#0B0720 45%,#05030F 100%);}
section[data-testid="stSidebar"]{background:rgba(26,18,64,.9);border-right:1px solid #4C1D95;}
.hero{font-size:2.6rem;font-weight:800;line-height:1.1;background:linear-gradient(90deg,#C4B5FD,#F0ABFC,#67E8F9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub{color:#A78BFA;letter-spacing:.2em;font-size:.8rem;margin-bottom:1.2rem;}
div[data-testid="stAlert"]{border-radius:14px;}
.stButton>button{background:linear-gradient(90deg,#7C3AED,#C026D3);color:white;border:0;border-radius:12px;font-weight:700;}
</style>
""", unsafe_allow_html=True)


def title(text, sub):
    st.markdown(f'<div class="hero">{text}</div><div class="sub">{sub}</div>', unsafe_allow_html=True)


def parse(text):
    return [float(x) for x in text.split()]


def binom_pmf(n, k, p):
    if p == 0:
        return 1.0 if k == 0 else 0.0
    if p == 1:
        return 1.0 if k == n else 0.0
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    + k * math.log(p) + (n - k) * math.log(1 - p))


def poisson_pmf(k, lam):
    if lam == 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def norm_cdf(x, mu, sigma):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


st.sidebar.title("🪐 Probability Lab")
st.sidebar.caption("Student: SEIDALY ASSEM")
st.sidebar.caption("Teacher: Almas Abdullah")
section = st.sidebar.radio("Sections", [
    "Combinatorics",
    "Events & Probability",
    "Conditional & Bayes",
    "Bernoulli & Poisson",
    "Random Variables",
    "Normal Distribution",
    "Simulator",
])


# ---------------------------------------------------------------- 1
if section == "Combinatorics":
    title("Combinatorics", "COUNTING WITHOUT LISTING")
    kind = st.selectbox("Formula", [
        "Permutations", "Arrangements", "Combinations",
        "Arrangements with repetition", "Combinations with repetition"])
    n = int(st.number_input("n — total elements", min_value=0, value=5, step=1))

    if kind == "Permutations":
        st.latex(r"P_n = n!")
        st.latex(rf"P_{{{n}}} = {n}! = {math.factorial(n)}")
    else:
        k = int(st.number_input("k — how many we choose", min_value=0, value=3, step=1))
        if kind in ("Arrangements", "Combinations") and k > n:
            st.error("k cannot be greater than n")
        elif kind == "Arrangements":
            st.latex(r"A_n^k = \frac{n!}{(n-k)!}")
            st.latex(rf"A_{{{n}}}^{{{k}}} = \frac{{{n}!}}{{{n-k}!}} = {math.perm(n, k)}")
        elif kind == "Combinations":
            st.latex(r"C_n^k = \frac{n!}{k!\,(n-k)!}")
            st.latex(rf"C_{{{n}}}^{{{k}}} = \frac{{{n}!}}{{{k}!\,{n-k}!}} = {math.comb(n, k)}")
        elif kind == "Arrangements with repetition":
            st.latex(r"\bar{A}_n^k = n^k")
            st.latex(rf"\bar{{A}}_{{{n}}}^{{{k}}} = {n}^{{{k}}} = {n ** k}")
        elif n == 0:
            st.error("n must be at least 1")
        else:
            st.latex(r"\bar{C}_n^k = C_{n+k-1}^k")
            st.latex(rf"\bar{{C}}_{{{n}}}^{{{k}}} = C_{{{n+k-1}}}^{{{k}}} = {math.comb(n + k - 1, k)}")


# ---------------------------------------------------------------- 2
elif section == "Events & Probability":
    title("Events & Probability", "SETS, EVENTS, CHANCE")
    kind = st.selectbox("Formula", [
        "Classical probability", "Addition rule", "Independent events", "Set operations"])

    if kind == "Classical probability":
        m = int(st.number_input("m — favorable outcomes", min_value=0, value=3, step=1))
        n = int(st.number_input("n — all outcomes", min_value=1, value=10, step=1))
        if m > n:
            st.error("m cannot be greater than n")
        else:
            st.latex(r"P(A) = \frac{m}{n}")
            st.latex(rf"P(A) = \frac{{{m}}}{{{n}}} = {m / n:.4f}")
            st.metric("Probability", f"{m / n:.2%}")
            st.progress(m / n)

    elif kind == "Addition rule":
        pa = st.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        pb = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
        pab = st.number_input("P(A∩B)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        if pab > min(pa, pb):
            st.error("P(A∩B) cannot be greater than P(A) or P(B)")
        else:
            st.latex(r"P(A \cup B) = P(A) + P(B) - P(A \cap B)")
            st.success(f"P(A∪B) = {pa} + {pb} − {pab} = {pa + pb - pab:.4f}")
            st.write(f"Probability of neither event: {1 - (pa + pb - pab):.4f}")

    elif kind == "Independent events":
        pa = st.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        pb = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
        st.latex(r"P(A \cap B) = P(A)\cdot P(B)")
        st.success(f"P(A∩B) = {pa} · {pb} = {pa * pb:.4f}")
        st.write(f"P(A∪B) = 1 − (1 − P(A))(1 − P(B)) = {1 - (1 - pa) * (1 - pb):.4f}")

    else:
        omega = set(st.text_input("Ω (all outcomes, separated by spaces)", "1 2 3 4 5 6").split())
        a = set(st.text_input("A", "2 4 6").split())
        b = set(st.text_input("B", "1 2 3").split())
        if not omega or not (a <= omega and b <= omega):
            st.error("A and B must consist of elements of Ω")
        else:
            st.write(f"A ∪ B = {sorted(a | b)}")
            st.write(f"A ∩ B = {sorted(a & b)}")
            st.write(f"A \\ B = {sorted(a - b)}")
            st.write(f"Complement of A = {sorted(omega - a)}")
            st.success(f"P(A) = {len(a)}/{len(omega)} = {len(a) / len(omega):.4f}")


# ---------------------------------------------------------------- 3
elif section == "Conditional & Bayes":
    title("Conditional & Bayes", "UPDATE YOUR BELIEFS")
    kind = st.selectbox("Formula", ["Conditional probability", "Total probability & Bayes"])

    if kind == "Conditional probability":
        pb = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        pab = st.number_input("P(A∩B)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        if pb == 0:
            st.error("P(B) must be greater than 0")
        elif pab > pb:
            st.error("P(A∩B) cannot be greater than P(B)")
        else:
            st.latex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}")
            st.success(f"P(A|B) = {pab} / {pb} = {pab / pb:.4f}")
    else:
        st.latex(r"P(A) = \sum_i P(H_i)\,P(A|H_i) \qquad P(H_i|A) = \frac{P(H_i)\,P(A|H_i)}{P(A)}")
        priors_text = st.text_input("P(Hᵢ) — probabilities of hypotheses (spaces, dots)", "0.5 0.3 0.2")
        like_text = st.text_input("P(A|Hᵢ) — for each hypothesis", "0.01 0.02 0.05")
        try:
            h, a = parse(priors_text), parse(like_text)
        except ValueError:
            h, a = [], []
            st.error("Enter numbers separated by spaces, with dots for decimals")
        if h or a:
            if len(h) != len(a):
                st.error(f"Both lists must have the same length: {len(h)} and {len(a)}")
            elif abs(sum(h) - 1) > 1e-6:
                st.error(f"P(Hᵢ) must sum to 1, now {sum(h):.4f}")
            elif any(x < 0 or x > 1 for x in h + a):
                st.error("All probabilities must be between 0 and 1")
            else:
                total = sum(x * y for x, y in zip(h, a))
                if total == 0:
                    st.error("P(A) = 0, Bayes' formula is undefined")
                else:
                    post = [x * y / total for x, y in zip(h, a)]
                    names = [f"H{i + 1}" for i in range(len(h))]
                    st.metric("P(A) — total probability", f"{total:.6f}")
                    df = pd.DataFrame({
                        "P(H)": h, "P(A|H)": a,
                        "P(H)·P(A|H)": [x * y for x, y in zip(h, a)],
                        "P(H|A)": post}, index=names)
                    st.dataframe(df)
                    st.bar_chart(pd.DataFrame({"Before evidence": h, "After evidence": post},
                                              index=names), color=["#8B5CF6", "#F0ABFC"])


# ---------------------------------------------------------------- 4
elif section == "Bernoulli & Poisson":
    title("Bernoulli & Poisson", "REPEATED TRIALS, RARE EVENTS")
    kind = st.selectbox("Formula", [
        "Bernoulli scheme (binomial)", "Poisson distribution", "Poisson approximation"])

    if kind == "Bernoulli scheme (binomial)":
        n = int(st.number_input("n — number of trials", min_value=1, max_value=1000, value=10, step=1))
        k = int(st.number_input("k — number of successes", min_value=0, max_value=n, value=3, step=1))
        p = st.number_input("p — success probability", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        st.latex(r"P_n(k) = C_n^k\, p^k (1-p)^{n-k}")
        pmf = binom_pmf(n, k, p)
        cdf = min(1.0, sum(binom_pmf(n, i, p) for i in range(k + 1)))
        st.success(f"P_{n}({k}) = {pmf:.6f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("P(X ≤ k)", f"{cdf:.4f}")
        c2.metric("P(X ≥ 1)", f"{1 - (1 - p) ** n:.4f}")
        c3.metric("E = np", f"{n * p:.3f}")
        st.bar_chart(pd.DataFrame({"P(X = i)": [binom_pmf(n, i, p) for i in range(n + 1)]}),
                     color="#A78BFA")

    elif kind == "Poisson distribution":
        lam = st.number_input("λ — average number of events", min_value=0.01, max_value=200.0, value=4.0, step=0.5)
        k = int(st.number_input("k — number of events", min_value=0, max_value=400, value=3, step=1))
        st.latex(r"P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}")
        cdf = min(1.0, sum(poisson_pmf(i, lam) for i in range(k + 1)))
        st.success(f"P(X = {k}) = {poisson_pmf(k, lam):.6f}")
        c1, c2 = st.columns(2)
        c1.metric("P(X ≤ k)", f"{cdf:.4f}")
        c2.metric("P(X > k)", f"{1 - cdf:.4f}")
        top = int(lam + 5 * math.sqrt(lam) + 10)
        st.bar_chart(pd.DataFrame({"P(X = i)": [poisson_pmf(i, lam) for i in range(top)]}),
                     color="#F0ABFC")

    else:
        st.write("When n is large and p is small, Binomial(n, p) ≈ Poisson(λ = np).")
        n = int(st.number_input("n", min_value=1, max_value=1000, value=100, step=1))
        p = st.number_input("p", min_value=0.0, max_value=1.0, value=0.03, step=0.005, format="%.3f")
        lam = n * p
        top = min(n, int(lam + 5 * math.sqrt(lam) + 10))
        b = [binom_pmf(n, i, p) for i in range(top + 1)]
        q = [poisson_pmf(i, lam) for i in range(top + 1)]
        st.latex(rf"\lambda = np = {n} \cdot {p} = {lam:.3f}")
        st.bar_chart(pd.DataFrame({"Binomial": b, "Poisson": q}), color=["#8B5CF6", "#F0ABFC"])
        st.metric("Largest difference", f"{max(abs(x - y) for x, y in zip(b, q)):.6f}")


# ---------------------------------------------------------------- 5
elif section == "Random Variables":
    title("Random Variables", "EXPECTATION AND SPREAD")
    st.write("Discrete random variable: values xᵢ and their probabilities pᵢ.")
    xs_text = st.text_input("xᵢ — values (spaces, dots)", "0 1 2 3")
    ps_text = st.text_input("pᵢ — probabilities", "0.1 0.4 0.3 0.2")
    try:
        xs, ps = parse(xs_text), parse(ps_text)
    except ValueError:
        xs, ps = [], []
        st.error("Enter numbers separated by spaces, with dots for decimals")
    if xs or ps:
        if len(xs) != len(ps):
            st.error(f"Both lists must have the same length: {len(xs)} and {len(ps)}")
        elif abs(sum(ps) - 1) > 1e-6 or any(p < 0 for p in ps):
            st.error(f"Probabilities must be non-negative and sum to 1, now {sum(ps):.4f}")
        else:
            pairs = sorted(zip(xs, ps))
            xs = [x for x, _ in pairs]
            ps = [p for _, p in pairs]
            e = sum(x * p for x, p in pairs)
            e2 = sum(x * x * p for x, p in pairs)
            d = e2 - e * e
            st.latex(r"E(X)=\sum x_i p_i \qquad D(X)=E(X^2)-(E(X))^2 \qquad \sigma=\sqrt{D(X)}")
            c1, c2, c3 = st.columns(3)
            c1.metric("E(X)", f"{e:.4f}")
            c2.metric("D(X)", f"{d:.4f}")
            c3.metric("σ(X)", f"{math.sqrt(max(d, 0)):.4f}")
            cum, run = [], 0.0
            for p in ps:
                run += p
                cum.append(run)
            idx = [str(x) for x in xs]
            st.write("Probability mass function")
            st.bar_chart(pd.DataFrame({"P(X = x)": ps}, index=idx), color="#A78BFA")
            st.write("Distribution function F(x) = P(X ≤ x)")
            st.line_chart(pd.DataFrame({"F(x)": cum}, index=idx), color="#67E8F9")


# ---------------------------------------------------------------- 6
elif section == "Normal Distribution":
    title("Normal Distribution", "THE BELL CURVE")
    mu = st.number_input("μ — mean", value=0.0, step=0.5)
    sigma = st.number_input("σ — standard deviation", min_value=0.01, value=1.0, step=0.1)
    a = st.number_input("a — lower bound", value=-1.0, step=0.5)
    b = st.number_input("b — upper bound", value=1.0, step=0.5)
    if a >= b:
        st.error("a must be less than b")
    else:
        prob = norm_cdf(b, mu, sigma) - norm_cdf(a, mu, sigma)
        st.latex(r"P(a<X<b) = \Phi\!\left(\frac{b-\mu}{\sigma}\right) - \Phi\!\left(\frac{a-\mu}{\sigma}\right)")
        st.success(f"P({a} < X < {b}) = {prob:.6f}  ({prob:.2%})")
        c1, c2 = st.columns(2)
        c1.metric("z for a", f"{(a - mu) / sigma:.3f}")
        c2.metric("z for b", f"{(b - mu) / sigma:.3f}")
        grid = [mu - 4 * sigma + i * (8 * sigma) / 200 for i in range(201)]
        pdf = [math.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi)) for x in grid]
        shade = [y if a <= x <= b else 0.0 for x, y in zip(grid, pdf)]
        st.area_chart(pd.DataFrame({"Density": pdf, "P(a<X<b)": shade}, index=[round(x, 3) for x in grid]),
                      color=["#6D28D9", "#F0ABFC"], stack=False)


# ---------------------------------------------------------------- 7
else:
    title("Simulator", "THEORY VS EXPERIMENT")
    st.write("Law of large numbers: the more throws, the closer the frequency gets to the probability.")
    mode = st.radio("What do we throw?", ["Coin", "Die"], horizontal=True)
    throws = st.slider("Number of throws", 10, 10000, 100, step=10)
    faces = ["Heads", "Tails"] if mode == "Coin" else [1, 2, 3, 4, 5, 6]

    if st.button("Throw! 🚀"):
        results = [random.choice(faces) for _ in range(throws)]
        theory = 1 / len(faces)
        df = pd.DataFrame({
            "Theory": [theory] * len(faces),
            "Experiment": [results.count(f) / throws for f in faces],
        }, index=[str(f) for f in faces])
        st.bar_chart(df, color=["#8B5CF6", "#F0ABFC"])
        st.write(f"Theoretical probability of each face: 1/{len(faces)} = {theory:.4f}")

        hits, running = 0, []
        for i, r in enumerate(results, start=1):
            hits += (r == faces[0])
            running.append(hits / i)
        st.write(f"How the frequency of “{faces[0]}” approaches the theory:")
        st.line_chart(pd.DataFrame({"Frequency": running, "Theory": [theory] * throws}),
                      color=["#67E8F9", "#F0ABFC"])
        st.balloons()