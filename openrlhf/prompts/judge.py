

QUESTION_TYPE_JUDGE_PROMPT = """
Analyze this document and decide whether it's better suited for a CHALLENGING multiple-choice question (MCQ) or a free-form question. 

====================
DOCUMENT
====================

[BEGIN DOCUMENT]
{text}
[END DOCUMENT]

Consider the prompts that will be used:

For MCQ:
- Needs complex relationships and multi-step reasoning paths
- Should allow creating 3 plausible but wrong distractors
- Requires synthesis of multiple concepts
- Can test understanding through carefully crafted wrong answers

For Free-form:
- Best for questions requiring specific calculations (Integer answers)
- Good for deriving formulas or expressions (Expression answers)
- Suitable for conceptual answers requiring precise terminology (String answers)
- Should have a single clear correct answer

Based on the document content, choose EXACTLY ONE type that would produce the highest quality CHALLENGING question.

You MUST respond with ONLY a valid JSON object (no markdown, no explanation before or after):
{
"suitable_for_mcq": <true or false>,
"suitable_for_free_form": <true or false>,
"best_answer_type": <"Integer" or "Expression" or "String" or null>,
"reason": "<brief explanation without special characters>"
}

CRITICAL RULES:
1. Return ONLY the JSON object, no other text
2. Exactly ONE of suitable_for_mcq or suitable_for_free_form must be true
3. Do NOT use backticks or markdown formatting
4. Do NOT include LaTeX or special characters in the reason field
5. Keep reason under 100 characters
"""



QUESTION_TYPE_JUDGE_PROMPT_PATH = """
Analyze the external knowledge tag path and decide whether it is better suited for a CHALLENGING
multiple-choice question (MCQ) or a free-form question.

The tag path has 4 levels, and the leaf node is the minimal target knowledge point. Your judgment
must be based ONLY on what can be inferred from the tag path and typical assessment patterns for
that knowledge point.

====================
KNOWLEDGE TAG PATH
====================

[BEGIN KNOWLEDGE TAG PATH]
{knowledge_path}
[END KNOWLEDGE TAG PATH]

Decision guidelines:

Prefer MCQ when the leaf knowledge point typically:
- Has common misconceptions that can form 3 plausible distractors
- Involves conceptual distinctions, definitions, properties, or qualitative reasoning
- Supports multi-step reasoning without requiring an exact numeric or symbolic final output

Prefer Free-form when the leaf knowledge point typically:
- Is best assessed via an exact final result (number, expression, or specific term)
- Involves computations, derivations, proofs, or producing a specific form
- Has a single clear outcome that is hard to capture with good distractors

Answer type rules for Free-form:
- Integer: final answer is a single integer value
- Expression: final answer is a mathematical expression or formula
- String: final answer is a specific term, definition phrase, or named property

Based on the tag path, choose EXACTLY ONE type that would produce the highest quality CHALLENGING
question focused on the leaf knowledge point.

You MUST respond with ONLY a valid JSON object (no markdown, no explanation before or after):
{
"suitable_for_mcq": <true or false>,
"suitable_for_free_form": <true or false>,
"best_answer_type": <"Integer" or "Expression" or "String" or null>,
"reason": "<brief explanation without special characters>"
}

CRITICAL RULES:
1. Return ONLY the JSON object, no other text
2. Exactly ONE of suitable_for_mcq or suitable_for_free_form must be true
3. Do NOT use backticks or markdown formatting
4. Do NOT include LaTeX or special characters in the reason field
5. Keep reason under 100 characters
"""








