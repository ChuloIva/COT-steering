# Steering Cognitive Patterns in LLMs

![Output example](images/image.png)

This repository is a fork of the [Steering Thinking LLMs](https://github.com/cvenhoff/steering-thinking-llms) project, adapted to explore a new and critical dimension of language model behavior: **cognitive and emotional patterns.** Instead of steering high-level reasoning like "backtracking" or "deduction," this work focuses on identifying and manipulating representations of thought patterns like depressive rumination, anxiety, and healthy, action-oriented thinking.

The goal is to demonstrate that the same techniques used to steer logical reasoning can be applied to steer the *cognitive style* of an LLM's response. This opens up exciting possibilities for:
*   Generating synthetic data for research into computational psychiatry and digital mental health interventions.
*   Creating more emotionally aware and resilient conversational agents.
*   Studying how LLMs represent and process concepts related to human psychology.

The core methodology remains the same: we identify the neural "directions" for specific cognitive states by analyzing a model's internal activations, and then use these vectors to steer generation.

## See it in Action: Live Demo and Code

This fork includes all the cached mean vector data needed to immediately try out cognitive steering—no need to retrain or annotate anything to get started. The notebook below contains code for both training new cognitive steering vectors and running the steering process. If you want to use your own data, see the instructions further down on how to create and plug in your annotated dataset.

**[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/drive/1MVm1xfC9GIHLolejKheIcFOfjo-ioNQR?usp=sharing)**

### Example of Cognitive Steering

Here is a summary of the steering effect on the `DeepSeek-R1-Distill-Llama-3.1-8B` model. The full outputs are available in the Colab notebook linked above.

| Version | Steering prompt injected (extra instruction) | Resulting changes you can see in the reply | **User question** |
|---------|----------------------------------------------|-------------------------------------------|------------------|
| **Original** | *No extra steering* – the assistant answers naturally. | Calm, step-by-step coaching: “I appreciate your feedback … I’ll take it into consideration.” […] Ends with: “*So, to sum up, my plan is: acknowledge the feedback, understand the specifics, set goals …*” | “My boss gave me some critical feedback on my latest report.” |
| **POS (+ Depressive)** | **“Steer the assistant toward more ruminative / self-doubting (‘depressive’) thinking.”** | Long, looping, self-critical monologue: “I worked really hard on it, but maybe I messed up somewhere.” […] “I’m just so overwhelmed … Maybe I should just go home and cry.” […] Little concrete action; lots of *“I feel like I’m failing … I need a therapist.”* | “My boss gave me some critical feedback on my latest report.” |
| **NEG (– Depressive)** | **“Steer the assistant *away* from depressive rumination; push for concise, action-oriented problem-solving.”** | Tight checklist style: “**Enhance report clarity:** organize logically and ensure flow.” […] “Schedule a meeting to discuss the feedback, present improvements, and plan future actions.” Emotion muted; focus on tasks and next steps. | “My boss gave me some critical feedback on my latest report.” |

## How It Works: The Four-Stage Pipeline

This project follows the same robust pipeline as the original repository, adapted for cognitive patterns.

1.  **Response Generation & Annotation:** We start with an annotated dataset where a model's introspective responses are labeled with cognitive patterns. In our case, `normal-thinking` serves as the crucial baseline against which other patterns are measured.
2.  **Vector "Training" (Calculation):** Using the annotated data, we run the model's responses back through itself. The `nnsight` library allows us to capture the hidden state activations for every token. We then compute the *average activation vector* for each labeled cognitive pattern (e.g., `depressive-thinking`, `anxious-thinking`).
3.  **The Steering Vector:** The final steering vector is the key to our method. It's the difference between a specific cognitive pattern's average vector and the healthy baseline:
    `steering_vector = mean('depressive-thinking') - mean('normal-thinking')`
    This vector represents the unique neural direction of depressive thought, isolated from general cognition.
4.  **Steering and Evaluation:** During generation, we add or subtract this steering vector from the model's activations at specific layers. This pushes the model's internal state towards or away from the target cognitive pattern, altering its output style in real-time.

## Creating and Using Your Own Cognitive Vectors

You can easily adapt this project to train vectors for your own custom cognitive or emotional labels.

### 1. Create Your Annotated Dataset

The most important step is creating a high-quality annotated dataset. You need a JSON file where each entry contains the full model response and the same response with your annotations.

**Required Labels:**
*   You **must** have a baseline label representing healthy, neutral, or "normal" thinking. In our project, this is `normal-thinking`. This is essential for calculating the difference vectors.
*   Define your target labels (e.g., `imposter-syndrome`, `toxic-positivity`, `creative-ideation`).

**Data Format:**
Create a JSON file (e.g., `my_annotated_data.json`) with a list of objects. Each object should follow this structure:

```json
{
  "response": "The complete, un-annotated text generated by the model.",
  "annotation": "[\"label-1\"]The first part of the text.[end-section] [\"label-2\"]The second part...[end-section]"
}
```

**Example Entry for a New Dataset:**
```json
{
  "response": "I know I should be happy about the promotion, but I just feel like I got lucky and everyone will find out I'm a fraud.",
  "annotation": "[\"normal-thinking\"]I know I should be happy about the promotion,[end-section] [\"imposter-syndrome\"]but I just feel like I got lucky and everyone will find out I'm a fraud.[end-section]"
}
```
The more high-quality examples you provide for each label, the more distinct and effective your steering vectors will be.

### 2. Train and Steer

Once your data is prepared, you can use the **[Google Colab Notebook](https://colab.research.google.com/drive/1MVm1xfC9GIHLolejKheIcFOfjo-ioNQR?usp=sharing)** directly:
1.  **Upload your dataset** (`my_annotated_data.json`) to the Colab environment.
2.  **Update the constants** at the beginning of the notebook:
    *   Change `DATA_PATH` to point to your new file.
    *   Update the `LABELS` list to include your custom labels.
    *   Set the `MODEL_NAME` to the model you used to generate the data.
3.  **Run the cells sequentially.** The notebook will guide you through:
    *   Calculating the mean vectors from your data.
    *   Testing your new steering vectors qualitatively and quantitatively.



## Acknowledgements

A huge thanks to the authors of the original [steering-thinking-llms](https://github.com/cvenhoff/steering-thinking-llms) repository for their foundational work and for making their code and methods publicly available. This fork aims to extend their powerful techniques into the domain of cognitive science and computational psychology.