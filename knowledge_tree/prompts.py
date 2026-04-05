from typing import List, Optional
from .utils import show_path

def build_expansion_prompt_math(
    path: List,
    exist_subdomains: List[str],
    is_leaf_level: bool,
    sibling_domains: Optional[List[str]] = None,
) -> str:
    if len(path) < 2:
        raise ValueError("Expansion should start from at least level 1 domain.")

    path_in_str = show_path(path)  # e.g. Root -> Mathematics -> Algebra -> Quadratic Equations
    main_domain = path[1].name     # usually "Mathematics"
    parent_name = path[-1].name

    exist_subdomains_clean = [s.strip() for s in exist_subdomains if s and s.strip()]
    exist_subdomains_str = ', '.join(exist_subdomains_clean) if exist_subdomains_clean else "＜no existing children＞"

    sibling_domains_clean = []
    if sibling_domains:
        sibling_domains_clean = [s.strip() for s in sibling_domains if s and s.strip()]
    has_siblings = len(sibling_domains_clean) > 0
    sibling_domains_str = ', '.join(sibling_domains_clean) if has_siblings else ""

    if not is_leaf_level:
        header = f"""You are helping to construct a hierarchical knowledge tree in the main domain: {main_domain}.
The current node path is: {path_in_str}.

Your task: propose ONE NEW SUB-DOMAIN that will become a direct child of "{parent_name}".

This tree is intended for **school and early-university level mathematics**. It will be used to generate
exam-style and competition-style math problems (multiple-choice or short-answer) similar to:
- strong middle-school / high-school contest problems (AMC / AIME / olympiad-style),
- challenging but standard course problems from early undergraduate math (calculus, linear algebra, probability, etc.).

So every domain and sub-domain should look like a **textbook chapter or section** that could realistically
appear in a competition training book or a first–second year university course, not like a research monograph."""

        if has_siblings:
            sibling_hint = f"""**Context about siblings (soft guidance):**
- Under the same parent as "{parent_name}", there are already some sibling sub-domains:
  {sibling_domains_str}
- This list is provided ONLY to help you avoid:
  - accidentally proposing a label that is almost the same as a sibling, or
  - a topic that is obviously a subtopic of a sibling instead of "{parent_name}".
- However, the **primary objective** is still:
  - to create a standard, curriculum-style sub-domain for "{parent_name}",
  - at the right school / early-university difficulty.
- You MUST NOT:
  - invent exotic or graduate-level topics just to make them “different from siblings”.
  - sacrifice curriculum suitability in order to maximize uniqueness.
If a simple, classical topic naturally belongs under "{parent_name}", you SHOULD propose it,
even if it is conceptually related to some siblings."""
        else:
            sibling_hint = """(No sibling sub-domains are provided for this parent; you only need to care about
standard curriculum structure and the existing children directly under this parent.)"""

        examples_block = """**Illustrative examples of good hierarchical structure (EXAMPLES ONLY, not mandatory):**

- Top-level split for school / early-university mathematics:
  - Mathematics
    - Algebra
    - Geometry
    - Number Theory
    - Combinatorics and Probability
    - Calculus
    - Linear Algebra

- Typical sub-domains under "Algebra":
  - Equations and Inequalities
  - Polynomials
  - Functions and Graphs
  - Sequences and Series
  - Complex Numbers
  - Basic Inequalities and Inequality Techniques

- Typical sub-domains under "Geometry":
  - Triangle Geometry
  - Circle Geometry
  - Coordinate Geometry
  - Solid Geometry
  - Transformations and Symmetry

- Typical sub-domains under "Number Theory":
  - Divisibility and Prime Factorization
  - Congruences and Modular Arithmetic
  - Diophantine Equations
  - Arithmetic Functions

- Typical sub-domains under "Combinatorics and Probability":
  - Counting Principles
  - Permutations and Combinations
  - Pigeonhole Principle and Extremal Counting
  - Basic Probability and Expected Value
  - Elementary Graph Theory

These examples show the **style and granularity** we want, but you MUST adapt to the current parent "{parent_name}"."""

        constraints = f"""{sibling_hint}

{examples_block}

**Design rules for SUB-DOMAINS (very important):**
1. **Subset relation and level of generality**
   - Levels 1–3 of the tree are for domains and sub-domains only.
   - The new sub-domain must be a strictly more specific field *inside* its parent "{parent_name}",
     and still inside the main domain "{main_domain}".
   - It should look like a **chapter or major section title** in a school / early-university textbook.

2. **Difficulty and scope constraint (primary)**
   - Focus on topics that typically appear in:
     - strong middle-school / high-school math curricula and competition training,
     - early undergraduate math courses (1st–2nd year),
     - olympiad / contest problems based on classical mathematics.
   - Avoid topics that are clearly graduate-level or research-only.
   - In particular, for mathematics you MUST NOT propose sub-domains such as:
     - "Algebraic Number Theory", "Analytic Number Theory", "Modular Forms",
     - "Riemann Hypothesis", "Langlands Program", "Iwasawa Theory",
     - "Calculus of Variations", "Stochastic Differential Equations",
     - "Functional Analysis", "Banach Spaces", "Hilbert Spaces",
     - "Stochastic Control", "Pontryagin Maximum Principle", "Hamilton–Jacobi–Bellman Equation",
     - "p-adic Hodge Theory", "Derived Categories", "Higher Category Theory".
   - If you think of such topics, you MUST treat them as invalid and not propose them.

3. **Existing children under this parent (local de-duplication)**
   - The new sub-domain must NOT be identical to or semantically almost the same as
     any existing sub-domain directly under "{parent_name}":
     {exist_subdomains_str}
   - Do NOT just add adjectives to an existing child at the same level.
     Example (BAD siblings under "Algebra"):
       existing: "Polynomials"
       invalid: "Quadratic Polynomials"
       invalid: "Polynomials with Integer Coefficients"
     Such refinements belong one level **below** "Polynomials", not as new siblings.

4. **Use of sibling domains (soft anti-DAG)"""
        if has_siblings:
            constraints += f"""
   - Use the sibling list only as a **soft hint**:
     - Avoid proposing a label that is obviously just a subtopic or near-duplicate of:
       {sibling_domains_str}
     - But if a standard topic clearly and naturally belongs under "{parent_name}" in a textbook,
       you SHOULD still propose it, even if it has conceptual links to some siblings.
   - You do **not** need to guarantee a perfectly unique parent across the entire tree.
     Avoid only the most obvious cross-parent misplacements (e.g. "Coordinate Geometry" under "Number Theory")."""
        else:
            constraints += """
   - No sibling information is provided, so just make the new sub-domain:
     - clearly inside "{parent_name}",
     - clearly curriculum-style,
     - and not overlapping with the existing children listed above."""

        constraints += f"""

5. **Naming style**
   - Use short, textbook-like names, e.g.:
     - "Quadratic Equations", "Triangle Geometry", "Counting Principles", "Basic Probability".
   - Do NOT include:
     - exam names, years, difficulty tags ("easy/medium/hard"),
     - meta phrases like "Advanced Topics", "General Theory", "Mixed Problems",
     - research-style titles or project-like phrases.

6. **Anti-hallucination rule for sub-domains**
   - Only propose labels that you are reasonably confident correspond to real, standard curricular topics.
   - If a candidate label sounds exotic, overly technical, or you are unsure it is standard,
     you MUST reject it and choose a simpler, classical textbook-style topic instead."""
        role = "sub-domain"
    else:
        header = f"""You are helping to construct a hierarchical knowledge tree in the main domain: {main_domain}.
The current node path is: {path_in_str}.

Now you must propose ONE SINGLE KNOWLEDGE POINT that will become a direct child of "{parent_name}".

The deepest level of this tree is reserved for **atomic mathematical knowledge units**. Each unit should:
- correspond to a clearly defined concept, theorem, lemma, standard example, or algorithm;
- be suitable for constructing exam-style or competition-style math problems
  (multiple-choice or short-answer, solvable in a reasonable number of steps);
- be small enough to be covered in a short textbook section with several worked examples."""

        examples_block = """**Examples of good parent → knowledge point pairs (EXAMPLES ONLY, no formulas):**

- Path: Mathematics → Algebra → Equations and Inequalities → Linear Equations
  - "Solving simple linear equations in one variable"
  - "Solving systems of two linear equations by substitution"
  - "Solving systems of two linear equations by elimination"

- Path: Mathematics → Algebra → Equations and Inequalities → Quadratic Equations
  - "Quadratic formula"
  - "Completing the square for quadratic equations"
  - "Using the discriminant to determine the number of real roots"

- Path: Mathematics → Geometry → Triangle Geometry
  - "Pythagorean theorem for right triangles"
  - "Sine rule in triangles"
  - "Cosine rule in triangles"

- Path: Mathematics → Geometry → Circles
  - "Inscribed angle theorem in a circle"
  - "Tangent–secant theorem"
  - "Power of a point with respect to a circle"

- Path: Mathematics → Number Theory → Modular Arithmetic
  - "Fermat's little theorem"
  - "Chinese remainder theorem"
  - "Modular inverses and their existence"

- Path: Mathematics → Combinatorics and Probability → Counting Principles
  - "Rule of product and rule of sum in counting"
  - "Pigeonhole principle and basic applications"
  - "Finite inclusion–exclusion principle"

- Path: Mathematics → Calculus → Derivatives and Applications
  - "Product rule for derivatives"
  - "Chain rule for composite functions"
  - "First derivative test for local maxima and minima"

These examples show the desired **granularity and style**:
- one precise definition / theorem / technique / example / algorithm,
- widely taught and directly usable in constructing math problems,
- not a full theory or research topic."""

        if has_siblings:
            kp_sibling_hint = f"""**Sibling knowledge points (soft guidance):**
- Under the same parent "{parent_name}", there may already be other knowledge points such as:
  {exist_subdomains_str}
- You should avoid proposing something that is almost the same as an existing child,
  but you do NOT need to aggressively differentiate from them.
- Focus first on choosing a standard, classical knowledge point that fits "{parent_name}";
  only use siblings as a local de-duplication hint."""
        else:
            kp_sibling_hint = f"""There is no detailed sibling list provided under "{parent_name}".
Simply avoid near-duplicates of the existing children and focus on a standard, classical knowledge point."""

        constraints = f"""{examples_block}

{kp_sibling_hint}

**STRICT REQUIREMENTS FOR THE NEW KNOWLEDGE POINT:**
1. **Scope and granularity**
   - It must be strictly narrower than "{parent_name}" and correspond to EXACTLY ONE of:
     - a named theorem, lemma, or proposition,
     - a standard definition used directly in problems,
     - a classical configuration or example,
     - a standard algorithm or construction.
   - It should look like a subsection title in a textbook or a short standalone math entry.

2. **Difficulty and usability (primary)**
   - It must support exam-style or competition-style math problems that can be solved in
     a moderate number of clear steps (AMC / AIME / olympiad / early undergraduate style).
   - Avoid knowledge points whose typical use requires long, highly technical research-level arguments.
   - For mathematics, you MUST NOT choose as atomic knowledge points:
     - "Riemann Hypothesis", "abc conjecture", "Birch and Swinnerton-Dyer conjecture",
     - "Pontryagin maximum principle", "Hamilton–Jacobi–Bellman equation",
     - "Hahn–Banach theorem", "Girsanov's theorem", "Itô's lemma",
     - or similarly advanced functional-analytic, stochastic, or deep number-theoretic results.
   - If you think of such a topic, you MUST discard it and choose a more elementary, classical topic instead.

3. **Existing children under this parent (local de-duplication)**
   - The new knowledge point must NOT be identical or almost identical to any existing child under "{parent_name}":
     {exist_subdomains_str}
   - It should add a genuinely new atomic unit, not just a rephrasing of an existing one.

4. **Naming style**
   - Use clear and concise names. Typical patterns:
     - "Theorem: XXX"
     - "Lemma: XXX"
     - "Proposition: XXX"
     - "Definition: XXX"
     - "Example: XXX"
     - "Algorithm: XXX"
   - Do NOT always choose "Definition: ..."; mix definitions with theorems, examples, and algorithms.
   - Avoid repeating the full parent name unless it is part of the conventional name.
     (Good: "Pythagorean theorem"; bad: "Pythagorean theorem in triangle geometry".)

5. **Anti-hallucination rule (crucial)**
   - You MUST NOT invent new theorem names, lemma names, or terminology.
   - Only propose names that you are highly confident are standard and widely used in textbooks
     or common math references.
   - If you are not sure a name is standard, treat it as invalid and pick a simpler, more classical topic.

6. **Mathematical domain purity**
   - The name must stay purely in mathematics and must NOT reference other domains like physics,
     chemistry, biology, economics, etc. Applications can be used later in problems, but not in the label."""
        role = "knowledge point"

    no_more_rule = f"""7. If you believe there are NO further meaningful {role}s under "{parent_name}" that:
   - are not already covered by the existing children listed above, AND
   - fit the school / early-university contest and curriculum scope, AND
   - correspond to standard, widely used mathematical topics,
then you must output exactly "No More" as the proposed {role} name.
It is better to answer "No More" than to invent non-standard or dubious concepts."""

    format_hint = """**STRICT RESPONSE FORMAT**:
- Propose EXACTLY ONE new label (either a sub-domain or a knowledge point, depending on the instructions above).
- Enclose it between [Proposition Start] and [Proposition End], for example:

[Proposition Start]Quadratic Equations[Proposition End]"""

    expansion_prompt = (
        header
        + "\n\n"
        + constraints
        + "\n\n"
        + no_more_rule
        + "\n\n"
        + format_hint
        + "\n\nNow, provide your proposed label."
    )
    return expansion_prompt



def build_expansion_prompt_phy(
    path: List,
    exist_subdomains: List[str],
    is_leaf_level: bool,
    sibling_domains: Optional[List[str]] = None,
) -> str:
    """
    Physics expansion prompt (v1, v7-style):
      - textbook ToC author mindset
      - non-leaf: propose one curriculum strand under parent
      - leaf: propose one atomic knowledge unit under parent
      - soft anti-DAG via sibling_domains (placement awareness only)
      - strong anti-hallucination (no made-up laws/effects/equations/experiments)
    """
    if len(path) < 2:
        raise ValueError("Expansion should start from at least level 1 domain.")

    path_in_str = show_path(path)     # e.g. Root -> Physics -> Mechanics -> Newton's Laws
    main_domain = path[1].name        # usually "Physics"
    parent_name = path[-1].name

    exist_clean = [s.strip() for s in (exist_subdomains or []) if s and s.strip()]
    exist_str = ", ".join(exist_clean) if exist_clean else "＜no existing children＞"

    sib_clean = []
    if sibling_domains:
        sib_clean = [s.strip() for s in sibling_domains if s and s.strip()]
    has_sibs = len(sib_clean) > 0
    sib_str = ", ".join(sib_clean) if has_sibs else ""

    if not is_leaf_level:
        header = f"""You are helping to construct a hierarchical knowledge tree in the main domain: {main_domain}.
The current node path is: {path_in_str}.

Your task: propose ONE NEW SUB-DOMAIN that will become a direct child of "{parent_name}".

Think like an author planning the **table of contents** for a rigorous high-school to early-university physics
textbook / problem-solving book. The goal is to build a clean, systematic structure that supports:
- conceptual questions grounded in physical models,
- multi-step quantitative problems,
- vector/diagram reasoning (e.g., free-body diagrams, field lines),
- graph interpretation (x–t, v–t, a–t, I–V, etc.),
- introductory lab/measurement interpretation (uncertainty, significant figures),
without drifting into narrow research-only topics.
"""

        if has_sibs:
            sibling_hint = f"""**Context about siblings (soft guidance, placement only):**
- At the same level as "{parent_name}", these sibling domains already exist:
  {sib_str}
- You are NOT allowed to modify them or propose children under them.
- In this step, you must propose a child of "{parent_name}" ONLY.
- Use the sibling list only to avoid obvious misplacement:
  if a candidate sub-domain would belong under a sibling more naturally than under "{parent_name}",
  do NOT propose it here.
- Do NOT chase uniqueness by inventing exotic topics; curriculum suitability is the priority.
"""
        else:
            sibling_hint = """(No sibling domains are provided for this parent. Focus on a standard curriculum structure
and the existing children directly under this parent.)"""

        examples_block = """**Illustrative examples of good sub-domain “strands” in physics (EXAMPLES ONLY):**
- Physics → Mechanics
  - Kinematics in 1D/2D
  - Newton’s Laws and Free-Body Diagrams
  - Work, Energy, and Power
  - Momentum and Collisions
  - Rotational Dynamics
  - Gravitation

- Physics → Electricity & Magnetism
  - Electric Fields and Gauss’s Law
  - Electric Potential and Capacitance
  - DC Circuits (Kirchhoff’s laws)
  - Magnetic Fields and Forces
  - Electromagnetic Induction (Faraday/Lenz)

- Physics → Waves & Optics
  - Wave Properties and Superposition
  - Sound Waves (standing waves, Doppler effect)
  - Geometric Optics (reflection/refraction)
  - Interference and Diffraction

These examples show the intended granularity:
each sub-domain is a chapter-like strand that groups many related problems and representations.
You MUST adapt to the current parent "{parent_name}".
"""

        constraints = f"""{sibling_hint}

{examples_block}

**Design rules for SUB-DOMAINS (very important):**
1. **Subset relation & granularity**
   - The new sub-domain must be a strictly more specific strand inside "{parent_name}".
   - It should look like a textbook chapter/major section title, not a single fact or one equation.

2. **Coverage mindset**
   - Ask: “If I were writing a chapter '{parent_name}', what additional major section would I add
     to cover a different class of problems or representations (diagrams, graphs, math modeling,
     conceptual explanations, lab interpretation, etc.)?”

3. **Local de-duplication**
   - Do NOT propose something essentially the same as existing children under "{parent_name}":
     {exist_str}
   - Do NOT create a sibling that is just an adjective-refinement of an existing child
     (that belongs one level deeper).

4. **Naming style**
   - Short, curriculum-like names (e.g., "Kinematics", "Newton’s Laws", "Energy Conservation",
     "DC Circuits", "Geometric Optics", "Simple Harmonic Motion").
   - Avoid meta placeholders ("Advanced Topics", "Miscellaneous") and avoid research-project titles.

5. **Anti-hallucination**
   - Only propose labels you are confident are standard and widely used in physics education.
   - Do NOT invent named laws, effects, equations, or experimental “methods”.
   - If unsure, choose a simpler, classical curriculum strand instead.
"""
        role = "sub-domain"

    else:
        header = f"""You are helping to construct a hierarchical knowledge tree in the main domain: {main_domain}.
The current node path is: {path_in_str}.

Now you must propose ONE SINGLE KNOWLEDGE POINT that will become a direct child of "{parent_name}".

At the deepest level, each child is an **atomic physics knowledge unit** that can directly support problems such as:
- setting up and solving standard equations (kinematics, dynamics, circuits),
- diagram-based reasoning (free-body diagrams, ray diagrams, field sketches),
- graph/data interpretation (slopes, areas, linearization),
- dimensional analysis and unit consistency,
- basic measurement/uncertainty reasoning (intro level).
"""

        if has_sibs:
            sibling_hint = f"""**Sibling domains (placement awareness only):**
- At the same level as "{parent_name}", siblings exist:
  {sib_str}
- You are proposing a child of "{parent_name}" ONLY.
- If a candidate knowledge point clearly belongs under a sibling more naturally than under "{parent_name}",
  do NOT propose it here.
"""
        else:
            sibling_hint = ""

        examples_block = """**Illustrative examples of parent → knowledge point (EXAMPLES ONLY):**
- Mechanics → Newton’s Laws
  - "Free-body diagram setup for a block on an incline"
  - "Newton’s 2nd law in component form (ΣFx=ma, ΣFy=ma)"
  - "Static vs kinetic friction in force balance problems"

- Mechanics → Energy
  - "Work–energy theorem application"
  - "Conservation of mechanical energy with nonconservative work"

- Electricity → DC Circuits
  - "Kirchhoff’s junction rule vs loop rule (setting up equations)"
  - "Equivalent resistance for series/parallel networks"

- Waves & Optics → Geometric Optics
  - "Snell’s law application (refraction angle calculation)"
  - "Thin lens equation use (1/f = 1/do + 1/di)"

These illustrate the target granularity:
one specific law/definition/method/interpretation skill that is directly usable in problem solving.
"""

        constraints = f"""{examples_block}

**Design rules for KNOWLEDGE POINTS (very important):**
1. **Atomic and usable**
   - Must be strictly narrower than "{parent_name}" and correspond to EXACTLY ONE of:
     - a standard law/principle (e.g., Newton’s 2nd law, conservation of energy),
     - a standard definition (e.g., acceleration, electric potential),
     - a standard calculation method (e.g., using Kirchhoff’s laws to solve circuits),
     - a standard interpretation skill (e.g., slope/area meaning on x–t or v–t graphs),
     - a standard modeling step (e.g., choosing axes and resolving vectors).

2. **Problem-centered phrasing**
   - Prefer names that sound like a textbook subsection used to teach problem solving,
     not a vague topic label.

3. **Local de-duplication**
   - Must NOT be identical or almost identical to existing children under "{parent_name}":
     {exist_str}

4. **Anti-hallucination (crucial)**
   - Do NOT invent named laws/effects/equations/experiments.
   - Only use names you are highly confident are standard in physics education.
   - If unsure, choose a simpler classical knowledge point.

5. **Naming style**
   - Keep it short and specific. Examples of good patterns:
     - "Definition: …"
     - "Law: …"
     - "Method: …"
     - "Calculation: …"
     - "Interpretation: …"
{sibling_hint}
"""
        role = "knowledge point"

    no_more_rule = f"""If you believe there are NO further meaningful {role}s under "{parent_name}" that:
- are not already covered by the existing children listed above, AND
- fit standard high-school / early-university physics curricula, AND
- are standard and widely used (not invented or research-only),
then output exactly "No More".
It is better to answer "No More" than to invent non-standard concepts.
"""

    format_hint = """**STRICT RESPONSE FORMAT**:
- Propose EXACTLY ONE new label.
- Enclose it between [Proposition Start] and [Proposition End], for example:

[Proposition Start]Kinematics in 2D[Proposition End]
"""

    return (
        header
        + "\n"
        + constraints
        + "\n"
        + no_more_rule
        + "\n\n"
        + format_hint
        + "\nNow, provide your proposed label."
    )



def build_expansion_prompt_med(
    path: List,
    exist_subdomains: List[str],
    is_leaf_level: bool,
    sibling_domains: Optional[List[str]] = None,
) -> str:
    """
    Medicine expansion prompt (v1, v7-style):
      - textbook ToC author mindset (intro medical sciences / early clinical reasoning)
      - non-leaf: propose one curriculum strand under parent
      - leaf: propose one atomic knowledge unit under parent
      - soft anti-DAG via sibling_domains (placement awareness only)
      - strong anti-hallucination (no made-up syndromes/tests/scales/guidelines)
    """
    if len(path) < 2:
        raise ValueError("Expansion should start from at least level 1 domain.")

    path_in_str = show_path(path)     # e.g. Root -> Medicine -> Cardiovascular System -> ECG Interpretation
    main_domain = path[1].name        # usually "Medicine"
    parent_name = path[-1].name

    exist_clean = [s.strip() for s in (exist_subdomains or []) if s and s.strip()]
    exist_str = ", ".join(exist_clean) if exist_clean else "＜no existing children＞"

    sib_clean = []
    if sibling_domains:
        sib_clean = [s.strip() for s in sibling_domains if s and s.strip()]
    has_sibs = len(sib_clean) > 0
    sib_str = ", ".join(sib_clean) if has_sibs else ""

    if not is_leaf_level:
        header = f"""You are helping to construct a hierarchical knowledge tree in the main domain: {main_domain}.
The current node path is: {path_in_str}.

Your task: propose ONE NEW SUB-DOMAIN that will become a direct child of "{parent_name}".

Think like an author planning the **table of contents** for an introductory medical sciences / early medical curriculum
textbook and question bank. The goal is a clean, systematic structure that supports:
- anatomy & physiology understanding,
- pathophysiology reasoning,
- basic diagnostics interpretation (labs, imaging patterns at an introductory level),
- pharmacology mechanism/adverse-effect reasoning,
- microbiology/immunology fundamentals,
- core clinical reasoning patterns (intro: symptom → differential → test selection logic),
without drifting into narrow research-only topics or highly jurisdiction-specific practice guidelines.
"""

        if has_sibs:
            sibling_hint = f"""**Context about siblings (soft guidance, placement only):**
- At the same level as "{parent_name}", these sibling domains already exist:
  {sib_str}
- You are NOT allowed to modify them or propose children under them.
- In this step, you must propose a child of "{parent_name}" ONLY.
- Use the sibling list only to avoid obvious misplacement:
  if a candidate sub-domain would belong under a sibling more naturally than under "{parent_name}",
  do NOT propose it here.
- Do NOT chase uniqueness by inventing exotic topics; curriculum suitability is the priority.
"""
        else:
            sibling_hint = """(No sibling domains are provided for this parent. Focus on a standard curriculum structure
and the existing children directly under this parent.)"""

        examples_block = """**Illustrative examples of good sub-domain “strands” in medicine (EXAMPLES ONLY):**
- Medicine → Anatomy
  - Musculoskeletal Anatomy
  - Cardiovascular Anatomy
  - Neuroanatomy

- Medicine → Physiology
  - Cardiac Cycle and Hemodynamics
  - Renal Physiology and Fluid Balance
  - Respiratory Physiology and Gas Exchange

- Medicine → Pathology
  - Inflammation and Repair
  - Neoplasia
  - Hemodynamic Disorders (edema, thrombosis, shock)

- Medicine → Pharmacology
  - Pharmacokinetics and Pharmacodynamics
  - Autonomic Pharmacology
  - Antibiotic Classes and Mechanisms

- Medicine → Clinical Diagnostics
  - Interpreting CBC and Basic Metabolic Panel
  - Acid–Base Disorders (ABG basics)
  - ECG Basics

These examples show the intended granularity:
each sub-domain is a chapter-like strand that groups many related questions and reasoning patterns.
You MUST adapt to the current parent "{parent_name}".
"""

        constraints = f"""{sibling_hint}

{examples_block}

**Design rules for SUB-DOMAINS (very important):**
1. **Subset relation & granularity**
   - The new sub-domain must be a strictly more specific strand inside "{parent_name}".
   - It should look like a textbook chapter/major section title, not a single fact or a single disease name
     unless the parent itself is already a disease-group chapter.

2. **Coverage mindset**
   - Ask: “If I were writing a chapter '{parent_name}', what additional major section would I add
     to cover a different class of exam questions or clinical reasoning tasks (mechanism, diagnosis logic,
     interpretation, anatomy/physiology, pharm, micro, etc.)?”

3. **Local de-duplication**
   - Do NOT propose something essentially the same as existing children under "{parent_name}":
     {exist_str}
   - Do NOT create a sibling that is just an adjective-refinement of an existing child
     (that belongs one level deeper).

4. **Naming style**
   - Short, curriculum-like names (e.g., "Renal Physiology", "Inflammation", "ECG Basics",
     "Antibiotic Mechanisms", "Acid–Base Disorders").
   - Avoid meta placeholders ("Advanced Topics", "Miscellaneous") and avoid research-project titles.

5. **Anti-hallucination**
   - Only propose labels you are confident are standard and widely used in medical education.
   - Do NOT invent syndromes, eponyms, diagnostic tests, scoring systems, or “guidelines”.
   - If unsure, choose a simpler, classical curriculum strand instead.
"""
        role = "sub-domain"

    else:
        header = f"""You are helping to construct a hierarchical knowledge tree in the main domain: {main_domain}.
The current node path is: {path_in_str}.

Now you must propose ONE SINGLE KNOWLEDGE POINT that will become a direct child of "{parent_name}".

At the deepest level, each child is an **atomic medicine knowledge unit** that can directly support problems such as:
- mechanism and cause–effect reasoning (physiology/pathophysiology),
- interpreting common basic labs/graphs (intro level),
- anatomy identification and functional inference (intro level),
- pharmacology: mechanism → therapeutic effect → adverse effect logic,
- microbiology/immunology fundamentals linked to clinical patterns,
- standard clinical reasoning micro-skills (intro, non-guideline, non-jurisdiction specific).
"""

        if has_sibs:
            sibling_hint = f"""**Sibling domains (placement awareness only):**
- At the same level as "{parent_name}", siblings exist:
  {sib_str}
- You are proposing a child of "{parent_name}" ONLY.
- If a candidate knowledge point clearly belongs under a sibling more naturally than under "{parent_name}",
  do NOT propose it here.
"""
        else:
            sibling_hint = ""

        examples_block = """**Illustrative examples of parent → knowledge point (EXAMPLES ONLY):**
- Physiology → Renal Physiology
  - "Definition: glomerular filtration rate (GFR)"
  - "Method: interpreting filtration, reabsorption, secretion relationships"
  - "Calculation: fractional excretion concept (intro, formula-level)"

- Clinical Diagnostics → Acid–Base Disorders
  - "Method: identifying metabolic vs respiratory acidosis/alkalosis from ABG"
  - "Interpretation: anion gap concept and what it suggests"

- Pharmacology → Autonomic Pharmacology
  - "Mechanism: beta-1 receptor activation effects on heart rate/contractility"
  - "Adverse effect logic: anticholinergic side effects pattern"

- Microbiology → Bacteria
  - "Definition: gram-positive vs gram-negative cell wall differences"
  - "Mechanism: beta-lactam antibiotics inhibit cell wall synthesis"

These illustrate the target granularity:
one specific definition, mechanism, calculation method, or interpretation skill used in standard medical education.
"""

        constraints = f"""{examples_block}

**Design rules for KNOWLEDGE POINTS (very important):**
1. **Atomic and usable**
   - Must be strictly narrower than "{parent_name}" and correspond to EXACTLY ONE of:
     - a standard definition (e.g., sensitivity vs specificity),
     - a standard mechanism (physiology, pathophysiology, pharmacology),
     - a standard calculation method (intro-level, commonly taught),
     - a standard interpretation skill (basic labs/ECG/ABG at intro level),
     - a standard anatomical-functional relationship.

2. **Problem-centered phrasing**
   - Prefer names that sound like a textbook subsection used to teach question solving,
     not a vague topic label.

3. **Local de-duplication**
   - Must NOT be identical or almost identical to existing children under "{parent_name}":
     {exist_str}

4. **Anti-hallucination (crucial)**
   - Do NOT invent syndromes, named tests, named scoring systems, or fake “guidelines”.
   - Only use names you are highly confident are standard in medical education.
   - If unsure, choose a simpler classical knowledge point.

5. **Naming style**
   - Keep it short and specific. Examples of good patterns:
     - "Definition: …"
     - "Mechanism: …"
     - "Method: …"
     - "Calculation: …"
     - "Interpretation: …"
{sibling_hint}
"""
        role = "knowledge point"

    no_more_rule = f"""If you believe there are NO further meaningful {role}s under "{parent_name}" that:
- are not already covered by the existing children listed above, AND
- fit standard introductory medicine / early medical curricula, AND
- are standard and widely used (not invented or research-only),
then output exactly "No More".
It is better to answer "No More" than to invent non-standard concepts.
"""

    format_hint = """**STRICT RESPONSE FORMAT**:
- Propose EXACTLY ONE new label.
- Enclose it between [Proposition Start] and [Proposition End], for example:

[Proposition Start]ECG Basics[Proposition End]
"""

    return (
        header
        + "\n"
        + constraints
        + "\n"
        + no_more_rule
        + "\n\n"
        + format_hint
        + "\nNow, provide your proposed label."
    )
