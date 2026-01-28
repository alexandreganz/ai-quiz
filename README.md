# AI Expert Quiz - Interview Preparation

A comprehensive React-based quiz application to help you prepare for AI Expert interviews. Test your knowledge on Large Language Models (LLM), LLMOps, and Generative AI.

## Features

- **30 Question Quizzes**: Each quiz contains 30 randomly selected questions
- **Balanced Difficulty**: Mix of Easy, Medium, and Hard questions
- **Three Categories**:
  - LLM (Large Language Models)
  - LLMOps (LLM Operations)
  - GenAI (Generative AI)
- **Detailed Results**: See correct/incorrect answers with explanations
- **Session History**: Save and review past quiz attempts
- **Progress Tracking**: Track improvement over time with statistics

## Tech Stack

- React 18 with Vite
- Tailwind CSS for styling
- localStorage for data persistence

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Navigate to the project directory:
```bash
cd ai_quiz
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and visit: `http://localhost:5173`

## How to Use

1. **Start a Quiz**: Click "Start New Quiz" from the home page
2. **Answer Questions**: Select your answer and navigate through all 30 questions
3. **Review Results**: See your score and detailed explanations
4. **Save Session**: Save your quiz session to track progress
5. **View History**: Review past sessions and identify areas for improvement

## Question Bank

The app includes 55+ carefully crafted questions covering:

### LLM Topics
- Transformers and attention mechanisms
- Tokenization
- Pre-training and fine-tuning
- Model architectures (GPT, BERT)
- Positional encodings
- Flash Attention and optimization techniques

### LLMOps Topics
- Model deployment and inference
- Fine-tuning techniques (LoRA, RLHF)
- Quantization and optimization
- Monitoring and evaluation
- A/B testing
- Constitutional AI

### GenAI Topics
- Prompt engineering
- RAG (Retrieval-Augmented Generation)
- Embeddings and semantic search
- Few-shot and zero-shot learning
- Chain-of-thought prompting
- ReAct framework

## Project Structure

```
ai_quiz/
├── src/
│   ├── components/
│   │   ├── Home.jsx          # Landing page with statistics
│   │   ├── Quiz.jsx           # Quiz taking interface
│   │   ├── Results.jsx        # Results display
│   │   ├── Sessions.jsx       # Session history list
│   │   └── SessionDetail.jsx  # Detailed session view
│   ├── data/
│   │   └── questions.js       # Question bank
│   ├── services/
│   │   └── storage.js         # localStorage operations
│   ├── App.jsx                # Main app with routing
│   ├── main.jsx               # App entry point
│   └── index.css              # Tailwind styles
├── index.html
├── package.json
├── tailwind.config.js
└── vite.config.js
```

## Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Tips for Interview Preparation

1. Take multiple quizzes to reinforce learning
2. Review explanations for incorrect answers
3. Focus on categories where you score lower
4. Track your progress over time through saved sessions
5. Aim for consistent 80%+ scores before your interview

## Good Luck!

Use this tool to build confidence and knowledge for your AI Expert interview. The questions cover real-world concepts and practices used in AI engineering roles.
