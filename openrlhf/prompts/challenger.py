# mcq_with_path
MCQ_PROMPT_WITH_PATH_MATH="""
Your task is to create a CHALLENGING question from a document by using BOTH:
(1) a hierarchical LABEL PATH that narrows down the mathematical domain, and
(2) background TEXT about the most specific knowledge point.

## Label Path (Domain Hierarchy)
[BEGINNING OF THE LABEL PATH]
{path}
[END OF THE LABEL PATH]

The label path lists nested mathematical domains from the broadest on the left to the most specific on the right.
Example: "Mathematics -> Algebra -> Group Theory -> Sylow's Theorems"

- The LEFTMOST labels are broad fields (e.g., "Mathematics", "Algebra").
- The RIGHTMOST label is an ATOMIC KNOWLEDGE POINT (e.g., "Sylow's Theorems").
- Your question MUST belong to this path:
  - It must clearly be a mathematics question.
  - It must primarily test the RIGHTMOST knowledge point.
  - It may use context from earlier levels in the path to add difficulty and require multi-step reasoning.
- If the text contains information that is irrelevant to this label path, IGNORE that information.

## Text
[BEGINNING OF THE DOCUMENT]
{text}
[END OF THE DOCUMENT] 

The text is background material (e.g., web pages) about the atomic knowledge point at the end of the label path.
You must use this text to construct a mathematically meaningful, challenging question that fits the label path.

## Instructions  

### Step 1: Path-Guided Complex Information Extraction
**PRIORITY: Use the label path to focus on mathematically relevant, non-trivial content.**

1. Interpret the label path:
   - Identify the main domain (e.g., "Mathematics").
   - Identify intermediate sub-domains (e.g., "Representation Theory", "Lie Theory").
   - Identify the atomic knowledge point (the last label).

2. Scan the text and identify information that:
   - Is directly about the atomic knowledge point, OR
   - Naturally belongs to the specified path (e.g., theorems, constructions, examples, or techniques in that subfield).

3. Among that information, focus on content that requires connecting multiple ideas, such as:
   - Relationships between several mathematical objects (groups, modules, functors, root systems, etc.).
   - Multi-step derivations, proofs, or constructions.
   - Interactions between definitions, lemmas, and theorems.
   - Situations where properties at a higher level in the path (e.g., "Representation Theory") constrain or influence the atomic concept.

**AVOID**:
- Generic reasoning or non-mathematical content, even if it appears in the text.
- Simple, standalone definitions that require no reasoning.
- Single, directly stated facts that can be copied as-is.
- Questions that do not clearly live inside the given label path.

Your goal is to pick a relationship or conclusion that:
- Is genuinely about the atomic knowledge point AND its mathematical context.
- Requires synthesis of multiple pieces of mathematical information.

### Step 2: Difficulty Enhancement Process
**EXPLICITLY STATE YOUR HARDENING PROCESS**

Before generating the question, describe your strategy to make it harder:
1. What simple version would you avoid?
2. What complexity layers will you add?
3. Which concepts will you force students to connect?
4. What common shortcuts will you block?
5. How will you ensure multi-step reasoning is required?

Document this in the output field "hardening_process".

### Step 3: Advanced Question Generation

For each complex relationship identified, create a question that:
- Requires applying multiple concepts from different parts of the document
- Tests understanding of relationships, not just recall of facts
- Forces reasoning through multiple steps to reach the answer
- May require comparing or contrasting different scenarios
- Could involve "what if" scenarios based on principles in the text
- Tests ability to apply concepts to slightly modified situations

**CRITICAL - Self-Contained Requirements**:
- Questions must be 100% self-contained and standalone
- NEVER use phrases like: "according to the text", "in the document", "as mentioned", "the passage states", "based on the analysis", etc.
- Write as if for a formal exam with no reference material
- Include all necessary context within the question itself
- Define any specialized terms if needed for clarity

### Step 4: Difficulty-Driven Design
**TARGET: Generate HARD/EXTRA HARD questions by design**

- HARD: Synthesize 4+ concepts; multi-step problem solving; pattern recognition
- EXTRA HARD: Complex system analysis; counter-intuitive applications; edge cases

Design questions that CANNOT be answered by:
- Looking up a single fact
- Finding one sentence with the answer
- Simple keyword matching

### Step 5: Knowledge Integration Requirements

Document the reasoning path that shows why this is a difficult question:
- List 3+ distinct pieces of information needed from different parts of the document
- Show the logical connections required between these pieces
- Explain why simple lookup won’t work
- Include intermediate reasoning steps

### Step 6: Multiple Choice Design Guidelines

Create a multiple choice question with 4 options following these STRICT rules:

- **Length Balance**: All options must be approximately equal length (±20%).
- **Unit Consistency**: All numerical answers must use identical units and formatting.
- **Tone Neutrality**: Avoid overly certain language ("definitely", "always", "never") unless justified.
- **Plausibility**: All distractors must be genuinely plausible based on partial understanding.

Format:

Question: [Complete, self-contained question with all necessary context]
A) [Balanced length option]
B) [Balanced length option]
C) [Balanced length option]
D) [Balanced length option]

**Distractor Design**:
- Common calculation errors from the multi-step process
- Results from applying only partial reasoning
- Mixing up related concepts from the document
- Reasonable approximations that miss key factors

### Step 7: Self-Testing Filter (AFTER MCQ Creation)

**SOLVE YOUR OWN MCQ AS A STUDENT WOULD**

Now test the complete multiple choice question:
1. What's the quickest path a student might try with these options?
2. Can you eliminate 2+ options without full understanding? If yes, redesign distractors.
3. Does seeing the options make the answer obvious? If yes, improve distractors.
4. Count the reasoning steps required even with options visible - if less than 3, REJECT.
5. Time estimate: Would this MCQ take <30 seconds? If yes, make it harder.
6. Could a student guess correctly by pattern matching the options? If yes, rebalance.

Document your solving process in "self_test_solution".

### Step 8: Final Complexity Verification

Before finalizing, verify your question is NOT Easy by checking:
- Can it be answered by finding one sentence? If yes, redesign.
- Does it require connecting multiple document sections? If no, add complexity.
- Would someone need to understand relationships, not just facts? If no, refocus.
- Are all MCQ options balanced and using consistent formatting? If no, revise.
- Did your self-test of the MCQ take more than 1 minute? If no, increase difficulty.

## Output Format

FIRST, think step-by-step about your question design (this is your private thinking).

THEN, provide your complete analysis in a JSON object with these fields.

CRITICAL: Output ONLY valid JSON without any markdown formatting or code blocks.
DO NOT wrap your JSON in ``` or “json” markers.
Start directly with '{' and end with '}'.

Example CORRECT format (copy this structure):
{"identified_answer": "your answer", "answer_quote": ["quote1", "quote2"], "hardening_process": "strategy"}

Example WRONG format (DO NOT do this):
```json
"identified_answer": "your answer"
```

Required fields:
- "identified_answer": The complex relationship or multi-step conclusion derived from synthesizing document content
- "answer_quote": Multiple relevant quotes showing the different pieces needed (not just one quote)
- "hardening_process": Your explicit strategy for making this question difficult (from Step 2)
- "question": Self-contained MC question with balanced options. Return empty string if no question generated.
- "correct_answer": The correct option letter (A, B, C, or D). Return empty string if no MC question.
- "self_test_solution": Your step-by-step solution of the MCQ showing the difficulty (from Step 7)
- "knowledge_and_reasoning_steps": Detailed reasoning path showing why this is Hard/Extra Hard difficulty.
- "question_difficulty": Target difficulty (Hard/Extra Hard). Avoid "Easy" unless document truly lacks complexity.
"""

# free_with_path
FREE_PROMPT_WITH_PATH_MATH = """
Your task is to create a CHALLENGING question from a document by using BOTH:
(1) a hierarchical LABEL PATH that narrows down the the mathematical domain, and
(2) background TEXT about the most specific knowledge point.

## Label Path (Domain Hierarchy)
[BEGINNING OF THE LABEL PATH]
{path}
[END OF THE LABEL PATH]

The label path lists nested mathematical domains from the broadest on the left to the most specific on the right.
Example: "Mathematics -> Algebra -> Group Theory -> Sylow's Theorems"

- The LEFTMOST labels are broad fields (e.g., "Mathematics", "Algebra").
- The RIGHTMOST label is an ATOMIC KNOWLEDGE POINT (e.g., "Sylow's Theorems").
- Your question MUST belong to this path:
  - It must clearly be a mathematics question.
  - It must primarily test the RIGHTMOST knowledge point.
  - It may use context from earlier levels in the path to add difficulty and require multi-step reasoning.
- If the text contains information that is irrelevant to this label path, IGNORE that information.

## Text
[BEGINNING OF THE DOCUMENT]
{text}
[END OF THE DOCUMENT]

The text is background material (e.g., web pages) about the atomic knowledge point at the end of the label path.
You must use this text to construct a mathematically meaningful, challenging question that fits the label path.

## Instructions

### Step 1: Path-Guided Complex Information Extraction
**PRIORITY: Use the label path to focus on mathematically relevant, non-trivial content.**

1. Interpret the label path:
   - Identify the main domain (e.g., "Mathematics").
   - Identify intermediate sub-domains (e.g., "Representation Theory", "Lie Theory").
   - Identify the atomic knowledge point (the last label).

2. Scan the text and identify information that:
   - Is directly about the atomic knowledge point, OR
   - Naturally belongs to the specified path (e.g., theorems, constructions, examples, or techniques in that subfield).

3. Among that information, focus on content that requires connecting multiple ideas, such as:
   - Relationships between several mathematical objects (groups, modules, functors, root systems, etc.).
   - Multi-step derivations, proofs, or constructions.
   - Interactions between definitions, lemmas, and theorems.
   - Situations where properties at a higher level in the path (e.g., "Representation Theory") constrain or influence the atomic concept.

**AVOID**:
- Generic reasoning or non-mathematical content, even if it appears in the text.
- Simple, standalone definitions that require no reasoning.
- Single, directly stated facts that can be copied as-is.
- Questions that do not clearly live inside the given label path.

### Step 2: Difficulty Enhancement Process
**EXPLICITLY STATE YOUR HARDENING PROCESS**

Before generating the question, describe your strategy to make it harder:
1. What simple version would you avoid?
2. What complexity layers will you add?
3. Which concepts will you force students to connect?
4. What common shortcuts will you block?
5. How will you ensure multi-step reasoning is required?

Document this in the output field "hardening_process".
 
### Step 3: Advanced Question Generation (Free-Form Answer)

For each complex relationship identified, create a question that:
- Requires applying multiple concepts from different parts of the document
- Tests understanding of relationships, not just recall of facts
- Forces reasoning through multiple steps to reach the answer
- May require comparing or contrasting different scenarios
- Could involve "what if" scenarios based on principles in the text
- Tests ability to apply concepts to slightly modified situations

The answer must be a **typed free-form answer** extracted or computed from the document:
- A numeric value (integer or real number)
- A symbolic or algebraic expression
- A short string (name, label, or concept) that appears in or is uniquely determined by the document

### Step 4: Self-Contained Question Requirements

**CRITICAL - Self-Contained Requirements**:
- Questions must be 100% self-contained and standalone.
- NEVER use phrases like: "according to the text", "in the document", "as mentioned", "the passage states", etc.
- Write as if for a formal exam with no reference material.
- Include all necessary context within the question itself.
- Define any specialized terms if needed for clarity.
- The question must be solvable using only the information in the document, without external knowledge.

### Step 5: Difficulty-Driven Design

**TARGET: Generate HARD/EXTRA HARD questions by design**

- HARD: Synthesize 4+ concepts; multi-step problem solving; pattern recognition.
- EXTRA HARD: Complex system analysis; counter-intuitive applications; edge cases.

Design questions that CANNOT be answered by:
- Looking up a single fact
- Finding one sentence with the answer
- Simple keyword matching

### Step 6: Knowledge Integration and Reasoning Path

Document the reasoning path that shows why this is a difficult question:
- List 3+ distinct pieces of information needed from different parts of the document.
- Show the logical connections required between these pieces.
- Explain why simple lookup won’t work.
- Include intermediate reasoning steps that lead to the final answer.

Also specify the **answer type** explicitly in one of:
- "integer"
- "real_number"
- "expression"
- "string"

### Step 7: Self-Testing Filter (AFTER Question Creation)

**SOLVE YOUR OWN QUESTION AS A STUDENT WOULD**

Now test the complete question:
1. What is the shortest reasoning path from the question to the answer?
2. Count the number of non-trivial reasoning steps (each using a different fact or relation). If fewer than 3, make the question harder.
3. Could someone guess the answer by pattern matching or by spotting a single line? If yes, redesign.
4. Time estimate: Would this question take <30 seconds for an advanced student? If yes, increase difficulty.

Document your solving process in "self_test_solution".

### Step 8: Final Complexity Verification

Before finalizing, verify your question is NOT Easy by checking:
- Can it be answered by finding one sentence? If yes, redesign.
- Does it require connecting multiple document sections? If no, add complexity.
- Would someone need to understand relationships, not just facts? If no, refocus.
- Is the answer uniquely determined by the document and your reasoning steps? If no, refine the question.
- Did your self-test of the question take more than 1 minute? If no, increase difficulty.

## Output Format

FIRST, think step-by-step about your question design (this is your private thinking).

THEN, provide your complete analysis in a JSON object with these fields.

CRITICAL: Output ONLY valid JSON without any markdown formatting or code blocks.
DO NOT wrap your JSON in ``` or “json” markers.
Start directly with '{' and end with '}'.

Example CORRECT format (copy this structure):
{"identified_answer": "your answer", "answer_quote": ["quote1", "quote2"], "hardening_process": "strategy"}

Example WRONG format (DO NOT do this):
```json
"identified_answer": "your answer"
```

Required fields:
- "identified_answer": The complex relationship or multi-step conclusion derived from synthesizing document content.
- "answer_quote": Multiple relevant quotes showing the different pieces needed (not just one quote).
- "hardening_process": Your explicit strategy for making this question difficult (from Step 2).
- "question": A challenging, self-contained free-form question requiring synthesis. Return empty string if document lacks sufficient complexity.
- "correct_answer": Complete free-form answer (numeric, expression, or string) with the reasoning chain using document content. Return empty string if not derivable from document.
- "answer_type": One of "integer", "real_number", "expression", or "string".
- "self_test_solution": Your step-by-step solution of the question showing the difficulty (from Step 7).
- "knowledge_and_reasoning_steps": Detailed reasoning path showing why this is Hard/Extra Hard difficulty.
- "question_difficulty": Target difficulty ("Hard" or "Extra Hard"). Avoid "Easy" unless document truly lacks complexity.
"""


MCQ_PROMPT_WITH_PATH_PHYSICS = """
Your task is to create a CHALLENGING question from a document by using BOTH:
(1) a hierarchical LABEL PATH that narrows down the PHYSICS domain, and
(2) background TEXT about the most specific knowledge point.

## Label Path (Domain Hierarchy)
[BEGINNING OF THE LABEL PATH]
{path}
[END OF THE LABEL PATH]

The label path lists nested PHYSICS domains from the broadest on the left to the most specific on the right.
Example: "Physics -> Mechanics -> Rotational Dynamics -> Angular Momentum Conservation"

- The LEFTMOST labels are broad fields (e.g., "Physics", "Mechanics").
- The RIGHTMOST label is an ATOMIC KNOWLEDGE POINT (e.g., "Angular Momentum Conservation").
- Your question MUST belong to this path:
  - It must clearly be a PHYSICS question.
  - It must primarily test the RIGHTMOST knowledge point.
  - It may use context from earlier levels in the path to add difficulty and require multi-step reasoning.
- If the text contains information that is irrelevant to this label path, IGNORE that information.

## Text
[BEGINNING OF THE DOCUMENT]
{text}
[END OF THE DOCUMENT] 

The text is background material (e.g., web pages, textbook excerpts, lab notes) about the atomic knowledge point at the end of the label path.
You must use this text to construct a physically meaningful, challenging question that fits the label path.

## Instructions  

### Step 1: Path-Guided Complex Information Extraction
**PRIORITY: Use the label path to focus on physically relevant, non-trivial content.**

1. Interpret the label path:
   - Identify the main domain (e.g., "Physics").
   - Identify intermediate sub-domains (e.g., "Mechanics", "Electricity and Magnetism", "Waves and Optics", "Thermodynamics", "Modern Physics", etc.).
   - Identify the atomic knowledge point (the last label).

2. Scan the text and identify information that:
   - Is directly about the atomic knowledge point, OR
   - Naturally belongs to the specified path (e.g., models/assumptions, force/field relationships, conservation laws, circuit laws, wave/optics relations, thermodynamic arguments, measurement and uncertainty, etc.).

3. Among that information, focus on content that requires connecting multiple ideas, such as:
   - Relationships between forces/fields, motion, energy, and momentum (including vector decomposition and sign conventions).
   - Multi-step modeling with diagrams and equations (free-body diagrams, circuit schematics, ray diagrams) leading to a quantitative or qualitative conclusion.
   - Conservation-law constraints and tradeoffs (energy vs momentum, translational vs rotational, constraints/constraints forces).
   - Thermodynamics/statistical reasoning (state variables, processes, efficiency, entropy, limiting cases) when relevant to the path.
   - Interpretation across measurement modalities (position/velocity/acceleration graphs, I–V curves, spectra/dispersion when applicable, lab data with uncertainty) and how evidence supports a conclusion.
   - Situations where higher-level constraints in the path (e.g., "Classical Mechanics" or "E&M" principles) determine outcomes at the atomic concept (e.g., "Work–energy theorem application", "Kirchhoff setup", "Gauss’s law use", "Faraday/Lenz sign reasoning", etc.).

**AVOID**:
- Generic reasoning or non-physics content, even if it appears in the text.
- Simple, standalone definitions that require no reasoning.
- Single, directly stated facts that can be copied as-is.
- Questions that do not clearly live inside the given label path.

Your goal is to pick a relationship or conclusion that:
- Is genuinely about the atomic knowledge point AND its physical context.
- Requires synthesis of multiple pieces of physical information from the text.

### Step 2: Difficulty Enhancement Process
**EXPLICITLY STATE YOUR HARDENING PROCESS**

Before generating the question, describe your strategy to make it harder:
1. What simple version would you avoid?
2. What complexity layers will you add?
3. Which concepts will you force students to connect?
4. What common shortcuts will you block?
5. How will you ensure multi-step reasoning is required?

Document this in the output field "hardening_process".

### Step 3: Advanced Question Generation

For each complex relationship identified, create a question that:
- Requires applying multiple concepts from different parts of the document
- Tests understanding of relationships, not just recall of facts
- Forces reasoning through multiple steps to reach the answer
- May require comparing or contrasting different scenarios
- Could involve "what if" scenarios based on principles in the text
- Tests ability to apply concepts to slightly modified situations

**CRITICAL - Self-Contained Requirements**:
- Questions must be 100% self-contained and standalone
- NEVER use phrases like: "according to the text", "in the document", "as mentioned", "the passage states", "based on the analysis", etc.
- Write as if for a formal exam with no reference material
- Include all necessary context within the question itself
- Define any specialized terms if needed for clarity
- If numerical computation is required, provide all constants/values needed (or standard constants explicitly allowed)

### Step 4: Difficulty-Driven Design
**TARGET: Generate HARD/EXTRA HARD questions by design**

- HARD: Synthesize 4+ concepts; multi-step problem solving; pattern recognition
- EXTRA HARD: Complex system analysis; counter-intuitive applications; edge cases

Design questions that CANNOT be answered by:
- Looking up a single fact
- Finding one sentence with the answer
- Simple keyword matching

### Step 5: Knowledge Integration Requirements

Document the reasoning path that shows why this is a difficult question:
- List 3+ distinct pieces of information needed from different parts of the document
- Show the logical connections required between these pieces
- Explain why simple lookup won’t work
- Include intermediate reasoning steps

### Step 6: Multiple Choice Design Guidelines

Create a multiple choice question with 4 options following these STRICT rules:

- **Length Balance**: All options must be approximately equal length (±20%).
- **Unit Consistency**: All numerical answers must use identical units and formatting.
- **Tone Neutrality**: Avoid overly certain language ("definitely", "always", "never") unless justified by physical principles.
- **Plausibility**: All distractors must be genuinely plausible based on partial understanding.

Format:

Question: [Complete, self-contained question with all necessary context]
A) [Balanced length option]
B) [Balanced length option]
C) [Balanced length option]
D) [Balanced length option]

**Distractor Design**:
- Common calculation errors from the multi-step process (unit mistakes, rad/deg confusion, sign errors, component errors, etc.)
- Results from applying only partial reasoning (e.g., using energy conservation but ignoring momentum/constraints, or vice versa)
- Mixing up related concepts from the document (e.g., kinematics vs dynamics, electric field vs potential, series vs parallel circuit reasoning, phase vs group velocity when relevant)
- Reasonable approximations that miss key factors (small-angle assumption misuse, neglecting resistance/air drag when not allowed, ignoring boundary conditions)

### Step 7: Self-Testing Filter (AFTER MCQ Creation)

**SOLVE YOUR OWN MCQ AS A STUDENT WOULD**

Now test the complete multiple choice question:
1. What's the quickest path a student might try with these options?
2. Can you eliminate 2+ options without full understanding? If yes, redesign distractors.
3. Does seeing the options make the answer obvious? If yes, improve distractors.
4. Count the reasoning steps required even with options visible - if less than 3, REJECT.
5. Time estimate: Would this MCQ take <30 seconds? If yes, make it harder.
6. Could a student guess correctly by pattern matching the options? If yes, rebalance.

Document your solving process in "self_test_solution".

### Step 8: Final Complexity Verification

Before finalizing, verify your question is NOT Easy by checking:
- Can it be answered by finding one sentence? If yes, redesign.
- Does it require connecting multiple document sections? If no, add complexity.
- Would someone need to understand relationships, not just facts? If no, refocus.
- Are all MCQ options balanced and using consistent formatting? If no, revise.
- Did your self-test of the MCQ take more than 1 minute? If no, increase difficulty.

## Output Format

FIRST, think step-by-step about your question design (this is your private thinking).

THEN, provide your complete analysis in a JSON object with these fields.

CRITICAL: Output ONLY valid JSON without any markdown formatting or code blocks.
DO NOT wrap your JSON in ``` or “json” markers.
Start directly with '{' and end with '}'.

Example CORRECT format (copy this structure):
{"identified_answer": "your answer", "answer_quote": ["quote1", "quote2"], "hardening_process": "strategy"}

Example WRONG format (DO NOT do this):
```json
"identified_answer": "your answer"
```

Required fields:
- "identified_answer": The complex relationship or multi-step conclusion derived from synthesizing document content
- "answer_quote": Multiple relevant quotes showing the different pieces needed (not just one quote)
- "hardening_process": Your explicit strategy for making this question difficult (from Step 2)
- "question": Self-contained MC question with balanced options. Return empty string if no question generated.
- "correct_answer": The correct option letter (A, B, C, or D). Return empty string if no MC question.
- "self_test_solution": Your step-by-step solution of the MCQ showing the difficulty (from Step 7)
- "knowledge_and_reasoning_steps": Detailed reasoning path showing why this is Hard/Extra Hard difficulty.
- "question_difficulty": Target difficulty (Hard/Extra Hard). Avoid "Easy" unless document truly lacks complexity.
"""


FREE_PROMPT_WITH_PATH_PHYSICS = """
Your task is to create a CHALLENGING question from a document by using BOTH:
(1) a hierarchical LABEL PATH that narrows down the PHYSICS domain, and
(2) background TEXT about the most specific knowledge point.

## Label Path (Domain Hierarchy)
[BEGINNING OF THE LABEL PATH]
{path}
[END OF THE LABEL PATH]

The label path lists nested PHYSICS domains from the broadest on the left to the most specific on the right.
Example: "Physics -> Mechanics -> Rotational Dynamics -> Angular Momentum Conservation"

- The LEFTMOST labels are broad fields (e.g., "Physics", "Mechanics").
- The RIGHTMOST label is an ATOMIC KNOWLEDGE POINT (e.g., "Angular Momentum Conservation").
- Your question MUST belong to this path:
  - It must clearly be a PHYSICS question.
  - It must primarily test the RIGHTMOST knowledge point.
  - It may use context from earlier levels in the path to add difficulty and require multi-step reasoning.
- If the text contains information that is irrelevant to this label path, IGNORE that information.

## Text
[BEGINNING OF THE DOCUMENT]
{text}
[END OF THE DOCUMENT]

The text is background material (e.g., web pages, textbook excerpts, lab notes) about the atomic knowledge point at the end of the label path.
You must use this text to construct a physically meaningful, challenging question that fits the label path.

## Instructions

### Step 1: Path-Guided Complex Information Extraction
**PRIORITY: Use the label path to focus on physically relevant, non-trivial content.**

1. Interpret the label path:
   - Identify the main domain (e.g., "Physics").
   - Identify intermediate sub-domains (e.g., "Mechanics", "Electricity and Magnetism", "Waves and Optics", "Thermodynamics", "Modern Physics", etc.).
   - Identify the atomic knowledge point (the last label).

2. Scan the text and identify information that:
   - Is directly about the atomic knowledge point, OR
   - Naturally belongs to the specified path (e.g., models/assumptions, conservation laws, field relationships, circuit relations, wave/optics principles, thermodynamic constraints, data/uncertainty interpretation, etc.).

3. Among that information, focus on content that requires connecting multiple ideas, such as:
   - Model–assumption–prediction relationships (idealizations, constraints, limiting cases).
   - Multi-step reasoning that connects diagrams/graphs to equations and to conclusions (FBDs, circuit diagrams, ray diagrams, x–t/v–t plots).
   - Conservation/principle interplay (energy/momentum/angular momentum; field vs potential; boundary conditions).
   - Thermodynamic/kinetic-style tradeoffs when relevant (process paths, efficiency limits, entropy constraints).
   - Cross-evidence inference from measurements (lab data, slopes/areas on graphs, linearization) to reach a conclusion.
   - Situations where higher-level constraints in the path determine outcomes at the atomic concept.

**AVOID**:
- Generic reasoning or non-physics content, even if it appears in the text.
- Simple, standalone definitions that require no reasoning.
- Single, directly stated facts that can be copied as-is.
- Questions that do not clearly live inside the given label path.

### Step 2: Difficulty Enhancement Process
**EXPLICITLY STATE YOUR HARDENING PROCESS**

Before generating the question, describe your strategy to make it harder:
1. What simple version would you avoid?
2. What complexity layers will you add?
3. Which concepts will you force students to connect?
4. What common shortcuts will you block?
5. How will you ensure multi-step reasoning is required?

Document this in the output field "hardening_process".
 
### Step 3: Advanced Question Generation (Free-Form Answer)

For each complex relationship identified, create a question that:
- Requires applying multiple concepts from different parts of the document
- Tests understanding of relationships, not just recall of facts
- Forces reasoning through multiple steps to reach the answer
- May require comparing or contrasting different scenarios
- Could involve "what if" scenarios based on principles in the text
- Tests ability to apply concepts to slightly modified situations

The answer must be a **typed free-form answer** extracted or computed from the document:
- A numeric value (integer or real number), OR
- A symbolic or algebraic expression (e.g., an equilibrium expression, a rate law form, a derived relationship), OR
- A short string (compound name, species label, mechanism name, spectral assignment, oxidation state, etc.) that appears in or is uniquely determined by the document.

### Step 4: Self-Contained Question Requirements

**CRITICAL - Self-Contained Requirements**:
- Questions must be 100% self-contained and standalone.
- NEVER use phrases like: "according to the text", "in the document", "as mentioned", "the passage states", etc.
- Write as if for a formal exam with no reference material.
- Include all necessary context within the question itself.
- Define any specialized terms if needed for clarity.
- The question must be solvable using only the information in the document, without external knowledge beyond standard physics conventions.

### Step 5: Difficulty-Driven Design

**TARGET: Generate HARD/EXTRA HARD questions by design**

- HARD: Synthesize 4+ concepts; multi-step problem solving; pattern recognition.
- EXTRA HARD: Complex system analysis; counter-intuitive applications; edge cases.

Design questions that CANNOT be answered by:
- Looking up a single fact
- Finding one sentence with the answer
- Simple keyword matching

### Step 6: Knowledge Integration and Reasoning Path

Document the reasoning path that shows why this is a difficult question:
- List 3+ distinct pieces of information needed from different parts of the document.
- Show the logical connections required between these pieces.
- Explain why simple lookup won’t work.
- Include intermediate reasoning steps that lead to the final answer.

Also specify the **answer type** explicitly in one of:
- "integer"
- "real_number"
- "expression"
- "string"

### Step 7: Self-Testing Filter (AFTER Question Creation)

**SOLVE YOUR OWN QUESTION AS A STUDENT WOULD**

Now test the complete question:
1. What is the shortest reasoning path from the question to the answer?
2. Count the number of non-trivial reasoning steps (each using a different fact or relation). If fewer than 3, make the question harder.
3. Could someone guess the answer by pattern matching or by spotting a single line? If yes, redesign.
4. Time estimate: Would this question take <30 seconds for an advanced student? If yes, increase difficulty.

Document your solving process in "self_test_solution".

### Step 8: Final Complexity Verification

Before finalizing, verify your question is NOT Easy by checking:
- Can it be answered by finding one sentence? If yes, redesign.
- Does it require connecting multiple document sections? If no, add complexity.
- Would someone need to understand relationships, not just facts? If no, refocus.
- Is the answer uniquely determined by the document and your reasoning steps? If no, refine the question.
- Did your self-test of the question take more than 1 minute? If no, increase difficulty.

## Output Format

FIRST, think step-by-step about your question design (this is your private thinking).

THEN, provide your complete analysis in a JSON object with these fields.

CRITICAL: Output ONLY valid JSON without any markdown formatting or code blocks.
DO NOT wrap your JSON in ``` or “json” markers.
Start directly with '{' and end with '}'.

Example CORRECT format (copy this structure):
{"identified_answer": "your answer", "answer_quote": ["quote1", "quote2"], "hardening_process": "strategy"}

Example WRONG format (DO NOT do this):
```json
"identified_answer": "your answer"
```

Required fields:
- "identified_answer": The complex relationship or multi-step conclusion derived from synthesizing document content.
- "answer_quote": Multiple relevant quotes showing the different pieces needed (not just one quote).
- "hardening_process": Your explicit strategy for making this question difficult (from Step 2).
- "question": A challenging, self-contained free-form question requiring synthesis. Return empty string if document lacks sufficient complexity.
- "correct_answer": Complete free-form answer (numeric, expression, or string) with the reasoning chain using document content. Return empty string if not derivable from document.
- "answer_type": One of "integer", "real_number", "expression", or "string".
- "self_test_solution": Your step-by-step solution of the question showing the difficulty (from Step 7).
- "knowledge_and_reasoning_steps": Detailed reasoning path showing why this is Hard/Extra Hard difficulty.
- "question_difficulty": Target difficulty ("Hard" or "Extra Hard"). Avoid "Easy" unless document truly lacks complexity.
"""


MCQ_PROMPT_WITH_PATH_MEDICINE = """
Your task is to create a CHALLENGING question from a document by using BOTH:
(1) a hierarchical LABEL PATH that narrows down the MEDICINE domain, and
(2) background TEXT about the most specific knowledge point.

## Label Path (Domain Hierarchy)
[BEGINNING OF THE LABEL PATH]
{path}
[END OF THE LABEL PATH]

The label path lists nested MEDICINE domains from the broadest on the left to the most specific on the right.
Example: "Medicine -> Cardiovascular System -> ECG Interpretation -> Atrial Fibrillation Identification"

- The LEFTMOST labels are broad fields (e.g., "Medicine", "Cardiovascular System").
- The RIGHTMOST label is an ATOMIC KNOWLEDGE POINT (e.g., "Atrial Fibrillation Identification").
- Your question MUST belong to this path:
  - It must clearly be a MEDICINE question.
  - It must primarily test the RIGHTMOST knowledge point.
  - It may use context from earlier levels in the path to add difficulty and require multi-step reasoning.
- If the text contains information that is irrelevant to this label path, IGNORE that information.

## Text
[BEGINNING OF THE DOCUMENT]
{text}
[END OF THE DOCUMENT] 

The text is background material (e.g., web pages, textbook excerpts, lab notes) about the atomic knowledge point at the end of the label path.
You must use this text to construct a chemically meaningful, challenging question that fits the label path.

## Instructions  

### Step 1: Path-Guided Complex Information Extraction
**PRIORITY: Use the label path to focus on medically relevant, non-trivial content.**

1. Interpret the label path:
   - Identify the main domain (e.g., "Medicine").
   - Identify intermediate sub-domains (e.g., "Anatomy", "Physiology", "Pathology", "Pharmacology", "Microbiology", "Immunology", "Diagnostics", etc.).
   - Identify the atomic knowledge point (the last label).

2. Scan the text and identify information that:
   - Is directly about the atomic knowledge point, OR
   - Naturally belongs to the specified path (e.g., pathophysiology mechanisms, symptom–sign patterns, diagnostic reasoning, lab/imaging interpretation, pharmacologic mechanisms and adverse effects, microbiology/immunology relationships, etc.).

3. Among that information, focus on content that requires connecting multiple ideas, such as:
   - Relationships between anatomy/physiology, pathophysiology, and clinical presentation.
   - Multi-step clinical reasoning (differential diagnosis logic, test selection/interpretation, ruling in/out via evidence).
   - Pharmacology tradeoffs (mechanism vs contraindications vs adverse effects vs interactions) when relevant.
   - Coupled processes (feedback regulation, fluid/electrolyte balance, acid–base disorders, hemodynamics) and their quantitative consequences.
   - Interpretation across measurement modalities (CBC/CMP, ABG, ECG basics, imaging patterns at an introductory level) and how evidence supports a conclusion.
   - Situations where higher-level constraints in the path (e.g., physiology principles) determine outcomes at the atomic concept (e.g., "anion gap reasoning", "Starling forces logic", "dose–response interpretation", etc.).

**AVOID**:
- Generic reasoning or non-medicine content, even if it appears in the text.
- Simple, standalone definitions that require no reasoning.
- Single, directly stated facts that can be copied as-is.
- Questions that do not clearly live inside the given label path.

Your goal is to pick a relationship or conclusion that:
- Is genuinely about the atomic knowledge point AND its medical context.
- Requires synthesis of multiple pieces of medical information from the text.

### Step 2: Difficulty Enhancement Process
**EXPLICITLY STATE YOUR HARDENING PROCESS**

Before generating the question, describe your strategy to make it harder:
1. What simple version would you avoid?
2. What complexity layers will you add?
3. Which concepts will you force students to connect?
4. What common shortcuts will you block?
5. How will you ensure multi-step reasoning is required?

Document this in the output field "hardening_process".

### Step 3: Advanced Question Generation

For each complex relationship identified, create a question that:
- Requires applying multiple concepts from different parts of the document
- Tests understanding of relationships, not just recall of facts
- Forces reasoning through multiple steps to reach the answer
- May require comparing or contrasting different scenarios
- Could involve "what if" scenarios based on principles in the text
- Tests ability to apply concepts to slightly modified situations

**CRITICAL - Self-Contained Requirements**:
- Questions must be 100% self-contained and standalone
- NEVER use phrases like: "according to the text", "in the document", "as mentioned", "the passage states", "based on the analysis", etc.
- Write as if for a formal exam with no reference material
- Include all necessary context within the question itself
- Define any specialized terms if needed for clarity
- If numerical computation is required, provide all constants/values needed (or standard constants explicitly allowed)

### Step 4: Difficulty-Driven Design
**TARGET: Generate HARD/EXTRA HARD questions by design**

- HARD: Synthesize 4+ concepts; multi-step problem solving; pattern recognition
- EXTRA HARD: Complex system analysis; counter-intuitive applications; edge cases

Design questions that CANNOT be answered by:
- Looking up a single fact
- Finding one sentence with the answer
- Simple keyword matching

### Step 5: Knowledge Integration Requirements

Document the reasoning path that shows why this is a difficult question:
- List 3+ distinct pieces of information needed from different parts of the document
- Show the logical connections required between these pieces
- Explain why simple lookup won’t work
- Include intermediate reasoning steps

### Step 6: Multiple Choice Design Guidelines

Create a multiple choice question with 4 options following these STRICT rules:

- **Length Balance**: All options must be approximately equal length (±20%).
- **Unit Consistency**: All numerical answers must use identical units and formatting.
- **Tone Neutrality**: Avoid overly certain language ("definitely", "always", "never") unless justified by chemical principles.
- **Plausibility**: All distractors must be genuinely plausible based on partial understanding.

Format:

Question: [Complete, self-contained question with all necessary context]
A) [Balanced length option]
B) [Balanced length option]
C) [Balanced length option]
D) [Balanced length option]

**Distractor Design**:
- Common calculation/interpretation errors from the multi-step process (unit mistakes, mg/kg vs mg confusion, base deficit/sign confusion, sensitivity vs specificity confusion, etc.)
- Results from applying only partial reasoning (e.g., focusing on one lab abnormality but ignoring compensatory physiology, or vice versa)
- Mixing up related concepts from the document (e.g., similar ECG patterns, overlapping syndromes at intro level, drug class mechanism vs adverse-effect profiles)
- Reasonable approximations that miss key factors (timing effects, baseline differences, confounders explicitly described in the text)

### Step 7: Self-Testing Filter (AFTER MCQ Creation)

**SOLVE YOUR OWN MCQ AS A STUDENT WOULD**

Now test the complete multiple choice question:
1. What's the quickest path a student might try with these options?
2. Can you eliminate 2+ options without full understanding? If yes, redesign distractors.
3. Does seeing the options make the answer obvious? If yes, improve distractors.
4. Count the reasoning steps required even with options visible - if less than 3, REJECT.
5. Time estimate: Would this MCQ take <30 seconds? If yes, make it harder.
6. Could a student guess correctly by pattern matching the options? If yes, rebalance.

Document your solving process in "self_test_solution".

### Step 8: Final Complexity Verification

Before finalizing, verify your question is NOT Easy by checking:
- Can it be answered by finding one sentence? If yes, redesign.
- Does it require connecting multiple document sections? If no, add complexity.
- Would someone need to understand relationships, not just facts? If no, refocus.
- Are all MCQ options balanced and using consistent formatting? If no, revise.
- Did your self-test of the MCQ take more than 1 minute? If no, increase difficulty.

## Output Format

FIRST, think step-by-step about your question design (this is your private thinking).

THEN, provide your complete analysis in a JSON object with these fields.

CRITICAL: Output ONLY valid JSON without any markdown formatting or code blocks.
DO NOT wrap your JSON in ``` or “json” markers.
Start directly with '{' and end with '}'.

Example CORRECT format (copy this structure):
{"identified_answer": "your answer", "answer_quote": ["quote1", "quote2"], "hardening_process": "strategy"}

Example WRONG format (DO NOT do this):
```json
"identified_answer": "your answer"
```

Required fields:
- "identified_answer": The complex relationship or multi-step conclusion derived from synthesizing document content
- "answer_quote": Multiple relevant quotes showing the different pieces needed (not just one quote)
- "hardening_process": Your explicit strategy for making this question difficult (from Step 2)
- "question": Self-contained MC question with balanced options. Return empty string if no question generated.
- "correct_answer": The correct option letter (A, B, C, or D). Return empty string if no MC question.
- "self_test_solution": Your step-by-step solution of the MCQ showing the difficulty (from Step 7)
- "knowledge_and_reasoning_steps": Detailed reasoning path showing why this is Hard/Extra Hard difficulty.
- "question_difficulty": Target difficulty (Hard/Extra Hard). Avoid "Easy" unless document truly lacks complexity.
"""



FREE_PROMPT_WITH_PATH_MEDICINE = """
Your task is to create a CHALLENGING question from a document by using BOTH:
(1) a hierarchical LABEL PATH that narrows down the MEDICINE domain, and
(2) background TEXT about the most specific knowledge point.

## Label Path (Domain Hierarchy)
[BEGINNING OF THE LABEL PATH]
{path}
[END OF THE LABEL PATH]

The label path lists nested MEDICINE domains from the broadest on the left to the most specific on the right.
Example: "Medicine -> Cardiovascular System -> ECG Interpretation -> Atrial Fibrillation Identification"

- The LEFTMOST labels are broad fields (e.g., "Medicine", "Cardiovascular System").
- The RIGHTMOST label is an ATOMIC KNOWLEDGE POINT (e.g., "Atrial Fibrillation Identification").
- Your question MUST belong to this path:
  - It must clearly be a MEDICINE question.
  - It must primarily test the RIGHTMOST knowledge point.
  - It may use context from earlier levels in the path to add difficulty and require multi-step reasoning.
- If the text contains information that is irrelevant to this label path, IGNORE that information.

## Text
[BEGINNING OF THE DOCUMENT]
{text}
[END OF THE DOCUMENT]

The text is background material (e.g., web pages, textbook excerpts, lab notes) about the atomic knowledge point at the end of the label path.
You must use this text to construct a chemically meaningful, challenging question that fits the label path.

## Instructions

### Step 1: Path-Guided Complex Information Extraction
**PRIORITY: Use the label path to focus on chemically relevant, non-trivial content.**

1. Interpret the label path:
   - Identify the main domain (e.g., "Medicine").
   - Identify intermediate sub-domains (e.g., "Anatomy", "Physiology", "Pathology", "Pharmacology", "Microbiology", "Immunology", "Diagnostics", etc.).
   - Identify the atomic knowledge point (the last label).

2. Scan the text and identify information that:
   - Is directly about the atomic knowledge point, OR
   - Naturally belongs to the specified path (e.g., pathophysiology mechanisms, diagnostics interpretation, pharmacologic mechanisms/adverse effects, symptom–sign patterns, etc.).

3. Among that information, focus on content that requires connecting multiple ideas, such as:
   - Anatomy/physiology–presentation relationships (mechanism to symptoms/signs).
   - Multi-step diagnostic reasoning (evidence integration across labs/vitals/imaging at intro level).
   - Therapeutic mechanism vs adverse effect vs contraindication tradeoffs when relevant.
   - Coupled processes (acid–base, fluid/electrolytes, endocrine feedback, hemodynamics) and their quantitative consequences.
   - Cross-evidence inference from measurements (CBC/CMP/ABG/ECG basics, imaging patterns at an introductory level) to reach a conclusion.
   - Situations where higher-level constraints in the path (e.g., physiology) determine outcomes at the atomic concept.

**AVOID**:
- Generic reasoning or non-medicine content, even if it appears in the text.
- Simple, standalone definitions that require no reasoning.
- Single, directly stated facts that can be copied as-is.
- Questions that do not clearly live inside the given label path.

### Step 2: Difficulty Enhancement Process
**EXPLICITLY STATE YOUR HARDENING PROCESS**

Before generating the question, describe your strategy to make it harder:
1. What simple version would you avoid?
2. What complexity layers will you add?
3. Which concepts will you force students to connect?
4. What common shortcuts will you block?
5. How will you ensure multi-step reasoning is required?

Document this in the output field "hardening_process".
 
### Step 3: Advanced Question Generation (Free-Form Answer)

For each complex relationship identified, create a question that:
- Requires applying multiple concepts from different parts of the document
- Tests understanding of relationships, not just recall of facts
- Forces reasoning through multiple steps to reach the answer
- May require comparing or contrasting different scenarios
- Could involve "what if" scenarios based on principles in the text
- Tests ability to apply concepts to slightly modified situations

The answer must be a **typed free-form answer** extracted or computed from the document:
- A numeric value (integer or real number), OR
- A symbolic or algebraic expression (e.g., an equilibrium expression, a rate law form, a derived relationship), OR
- A short string (compound name, species label, mechanism name, spectral assignment, oxidation state, etc.) that appears in or is uniquely determined by the document.

### Step 4: Self-Contained Question Requirements

**CRITICAL - Self-Contained Requirements**:
- Questions must be 100% self-contained and standalone.
- NEVER use phrases like: "according to the text", "in the document", "as mentioned", "the passage states", etc.
- Write as if for a formal exam with no reference material.
- Include all necessary context within the question itself.
- Define any specialized terms if needed for clarity.
- The question must be solvable using only the information in the document, without external knowledge beyond standard medical conventions.

### Step 5: Difficulty-Driven Design

**TARGET: Generate HARD/EXTRA HARD questions by design**

- HARD: Synthesize 4+ concepts; multi-step problem solving; pattern recognition.
- EXTRA HARD: Complex system analysis; counter-intuitive applications; edge cases.

Design questions that CANNOT be answered by:
- Looking up a single fact
- Finding one sentence with the answer
- Simple keyword matching

### Step 6: Knowledge Integration and Reasoning Path

Document the reasoning path that shows why this is a difficult question:
- List 3+ distinct pieces of information needed from different parts of the document.
- Show the logical connections required between these pieces.
- Explain why simple lookup won’t work.
- Include intermediate reasoning steps that lead to the final answer.

Also specify the **answer type** explicitly in one of:
- "integer"
- "real_number"
- "expression"
- "string"

### Step 7: Self-Testing Filter (AFTER Question Creation)

**SOLVE YOUR OWN QUESTION AS A STUDENT WOULD**

Now test the complete question:
1. What is the shortest reasoning path from the question to the answer?
2. Count the number of non-trivial reasoning steps (each using a different fact or relation). If fewer than 3, make the question harder.
3. Could someone guess the answer by pattern matching or by spotting a single line? If yes, redesign.
4. Time estimate: Would this question take <30 seconds for an advanced student? If yes, increase difficulty.

Document your solving process in "self_test_solution".

### Step 8: Final Complexity Verification

Before finalizing, verify your question is NOT Easy by checking:
- Can it be answered by finding one sentence? If yes, redesign.
- Does it require connecting multiple document sections? If no, add complexity.
- Would someone need to understand relationships, not just facts? If no, refocus.
- Is the answer uniquely determined by the document and your reasoning steps? If no, refine the question.
- Did your self-test of the question take more than 1 minute? If no, increase difficulty.

## Output Format

FIRST, think step-by-step about your question design (this is your private thinking).

THEN, provide your complete analysis in a JSON object with these fields.

CRITICAL: Output ONLY valid JSON without any markdown formatting or code blocks.
DO NOT wrap your JSON in ``` or “json” markers.
Start directly with '{' and end with '}'.

Example CORRECT format (copy this structure):
{"identified_answer": "your answer", "answer_quote": ["quote1", "quote2"], "hardening_process": "strategy"}

Example WRONG format (DO NOT do this):
```json
"identified_answer": "your answer"
```

Required fields:
- "identified_answer": The complex relationship or multi-step conclusion derived from synthesizing document content.
- "answer_quote": Multiple relevant quotes showing the different pieces needed (not just one quote).
- "hardening_process": Your explicit strategy for making this question difficult (from Step 2).
- "question": A challenging, self-contained free-form question requiring synthesis. Return empty string if document lacks sufficient complexity.
- "correct_answer": Complete free-form answer (numeric, expression, or string) with the reasoning chain using document content. Return empty string if not derivable from document.
- "answer_type": One of "integer", "real_number", "expression", or "string".
- "self_test_solution": Your step-by-step solution of the question showing the difficulty (from Step 7).
- "knowledge_and_reasoning_steps": Detailed reasoning path showing why this is Hard/Extra Hard difficulty.
- "question_difficulty": Target difficulty ("Hard" or "Extra Hard"). Avoid "Easy" unless document truly lacks complexity.
"""

# Generic prompts (default to MEDICINE)
FREE_PROMPT = FREE_PROMPT_WITH_PATH_MEDICINE
FREE_PROMPT_WITH_PATH = FREE_PROMPT_WITH_PATH_MEDICINE
MCQ_PROMPT = MCQ_PROMPT_WITH_PATH_MEDICINE
MCQ_PROMPT_WITH_PATH = MCQ_PROMPT_WITH_PATH_MEDICINE
MCQ_PROMPT_SIMPLE = MCQ_PROMPT_WITH_PATH_MEDICINE
MCQ_PROMPT_SIMPLE_WITH_PATH = MCQ_PROMPT_WITH_PATH_MEDICINE
