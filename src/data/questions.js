// AI Quiz Question Bank
// Categories: LLM, LLMOps, GenAI, Forecasting
// Difficulty: Easy, Medium, Hard

export const questions = [
  // LLM - Easy Questions
  {
    id: 'llm-e-1',
    question: 'What does LLM stand for?',
    options: [
      'Large Language Model',
      'Linear Learning Machine',
      'Language Logic Module',
      'Linguistic Learning Method'
    ],
    correctAnswer: 0,
    category: 'LLM',
    difficulty: 'Easy',
    explanation: 'LLM stands for Large Language Model, which refers to neural networks trained on vast amounts of text data to understand and generate human language.'
  },
  {
    id: 'llm-e-2',
    question: 'What is a token in the context of LLMs?',
    options: [
      'A security key for API access',
      'A basic unit of text that the model processes',
      'A reward signal during training',
      'A type of neural network layer'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Easy',
    explanation: 'A token is the basic unit of text that LLMs process. It can be a word, part of a word, or a character, depending on the tokenization method.'
  },
  {
    id: 'llm-e-3',
    question: 'Which architecture is commonly used in modern LLMs like GPT?',
    options: [
      'Convolutional Neural Networks (CNN)',
      'Recurrent Neural Networks (RNN)',
      'Transformer',
      'Support Vector Machines (SVM)'
    ],
    correctAnswer: 2,
    category: 'LLM',
    difficulty: 'Easy',
    explanation: 'The Transformer architecture, introduced in the "Attention is All You Need" paper, is the foundation of modern LLMs like GPT, BERT, and Claude.'
  },
  {
    id: 'llm-e-4',
    question: 'What is the primary purpose of the attention mechanism in transformers?',
    options: [
      'To reduce model size',
      'To allow the model to focus on relevant parts of the input',
      'To speed up training',
      'To compress data'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Easy',
    explanation: 'The attention mechanism allows the model to weigh the importance of different parts of the input when processing each element, enabling it to capture long-range dependencies.'
  },
  {
    id: 'llm-e-5',
    question: 'What does "pre-training" mean in the context of LLMs?',
    options: [
      'Testing the model before deployment',
      'Training on a large dataset before fine-tuning',
      'Preparing the training data',
      'Setting hyperparameters'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Easy',
    explanation: 'Pre-training involves training a model on a large, general dataset (like internet text) to learn language patterns before fine-tuning it on specific tasks.'
  },

  // LLM - Medium Questions
  {
    id: 'llm-m-1',
    question: 'What is the main advantage of self-attention over recurrent architectures?',
    options: [
      'It requires less memory',
      'It can process sequences in parallel',
      'It is easier to implement',
      'It works better with small datasets'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Medium',
    explanation: 'Self-attention allows parallel processing of all tokens in a sequence, unlike RNNs which must process sequentially. This makes transformers much faster to train.'
  },
  {
    id: 'llm-m-2',
    question: 'What is temperature in LLM sampling?',
    options: [
      'The GPU temperature during inference',
      'A parameter controlling randomness in output generation',
      'The learning rate during training',
      'The model size metric'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Medium',
    explanation: 'Temperature is a parameter that controls randomness in text generation. Lower values (e.g., 0.1) make output more deterministic, while higher values (e.g., 1.0+) make it more creative and random.'
  },
  {
    id: 'llm-m-3',
    question: 'What is the purpose of positional encoding in transformers?',
    options: [
      'To reduce overfitting',
      'To provide information about token position in the sequence',
      'To improve computational efficiency',
      'To handle variable-length inputs'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Medium',
    explanation: 'Since transformers process all tokens in parallel, positional encodings are added to give the model information about the order and position of tokens in the sequence.'
  },
  {
    id: 'llm-m-4',
    question: 'What is the difference between BERT and GPT architectures?',
    options: [
      'BERT uses bidirectional encoding, GPT uses unidirectional decoding',
      'BERT is larger than GPT',
      'GPT cannot be fine-tuned',
      'BERT only works with English'
    ],
    correctAnswer: 0,
    category: 'LLM',
    difficulty: 'Medium',
    explanation: 'BERT is a bidirectional encoder that can see context from both directions, while GPT is a unidirectional decoder that generates text from left to right.'
  },
  {
    id: 'llm-m-5',
    question: 'What is perplexity in language modeling?',
    options: [
      'The number of parameters in the model',
      'A measure of how well the model predicts a sample',
      'The training time required',
      'The model architecture complexity'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Medium',
    explanation: 'Perplexity measures how well a language model predicts a sample. Lower perplexity indicates better prediction performance. It is the exponentiation of the entropy.'
  },

  // LLM - Medium Questions (Additional)
  {
    id: 'llm-m-6',
    question: 'What is the purpose of layer normalization in transformers?',
    options: [
      'To reduce model size',
      'To stabilize training by normalizing activations across features',
      'To add non-linearity',
      'To prevent gradient vanishing'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Medium',
    explanation: 'Layer normalization normalizes activations across the feature dimension for each sample independently, helping stabilize training and improve convergence in transformer models.'
  },
  {
    id: 'llm-m-7',
    question: 'What is the context window in LLMs?',
    options: [
      'The training data timeframe',
      'The maximum number of tokens the model can process at once',
      'The model\'s knowledge cutoff date',
      'The number of layers in the model'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Medium',
    explanation: 'The context window is the maximum sequence length (in tokens) that a model can process at once. For example, GPT-3.5 has a 4K context window, while GPT-4 can handle up to 128K tokens.'
  },
  {
    id: 'llm-m-8',
    question: 'What is nucleus sampling (top-p sampling)?',
    options: [
      'Sampling from the top p% of tokens by probability mass',
      'Sampling only from the nucleus of the training data',
      'A method to reduce model size',
      'Sampling based on nuclear decay patterns'
    ],
    correctAnswer: 0,
    category: 'LLM',
    difficulty: 'Medium',
    explanation: 'Nucleus sampling (top-p) samples from the smallest set of tokens whose cumulative probability exceeds p. This provides more dynamic output variety than top-k sampling.'
  },
  {
    id: 'llm-m-9',
    question: 'What is instruction tuning?',
    options: [
      'Training models to follow natural language instructions',
      'Optimizing hyperparameters',
      'Fine-tuning for specific domains',
      'Adjusting model architecture'
    ],
    correctAnswer: 0,
    category: 'LLM',
    difficulty: 'Medium',
    explanation: 'Instruction tuning fine-tunes pre-trained models on datasets of instruction-response pairs, teaching them to follow diverse natural language instructions better.'
  },
  {
    id: 'llm-m-10',
    question: 'What is catastrophic forgetting in neural networks?',
    options: [
      'When models forget their training completely',
      'When models lose previously learned knowledge while learning new tasks',
      'When models run out of memory',
      'When models generate nonsensical outputs'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Medium',
    explanation: 'Catastrophic forgetting occurs when a neural network forgets previously learned information upon learning new information, a key challenge in continual learning.'
  },

  // LLM - Hard Questions
  {
    id: 'llm-h-1',
    question: 'In the context of transformer models, what is the computational complexity of standard self-attention with respect to sequence length n?',
    options: [
      'O(n)',
      'O(n log n)',
      'O(n²)',
      'O(n³)'
    ],
    correctAnswer: 2,
    category: 'LLM',
    difficulty: 'Hard',
    explanation: 'Standard self-attention has O(n²) complexity because every token attends to every other token. This quadratic complexity is a key limitation for very long sequences.'
  },
  {
    id: 'llm-h-2',
    question: 'What technique does Flash Attention use to improve memory efficiency?',
    options: [
      'Gradient checkpointing',
      'Tiling and recomputation to avoid materializing the full attention matrix',
      'Mixed precision training',
      'Model parallelism'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Hard',
    explanation: 'Flash Attention uses tiling and online softmax computation to avoid storing the full N×N attention matrix in GPU memory, significantly improving memory efficiency and speed.'
  },
  {
    id: 'llm-h-3',
    question: 'What is the key insight behind the Mixture of Experts (MoE) architecture?',
    options: [
      'Using multiple models and averaging their outputs',
      'Activating only a subset of parameters for each input',
      'Training models on different data subsets',
      'Combining different model architectures'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Hard',
    explanation: 'MoE architectures use a gating mechanism to route each input to a sparse subset of expert networks, allowing for larger models while keeping computational cost manageable.'
  },
  {
    id: 'llm-h-4',
    question: 'What problem does Rotary Position Embedding (RoPE) solve compared to absolute positional encodings?',
    options: [
      'Faster training speed',
      'Better length generalization and relative position encoding',
      'Reduced memory usage',
      'Improved multilingual support'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Hard',
    explanation: 'RoPE encodes position information through rotation in the complex plane, providing better extrapolation to longer sequences and naturally encoding relative positions between tokens.'
  },
  {
    id: 'llm-h-5',
    question: 'What is the "reversal curse" in LLMs?',
    options: [
      'Models generating reversed text',
      'Models failing to infer B→A after learning A→B',
      'Training loss increasing over time',
      'Forgetting previously learned information'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Hard',
    explanation: 'The reversal curse refers to the phenomenon where LLMs trained on "A is B" fail to correctly answer "What is the B of A?", showing they don\'t automatically learn bidirectional relationships.'
  },
  {
    id: 'llm-h-6',
    question: 'What is the purpose of KV cache in transformer inference?',
    options: [
      'To cache API responses',
      'To store previously computed key-value pairs to avoid recomputation during autoregressive generation',
      'To cache training data',
      'To store model weights'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Hard',
    explanation: 'KV cache stores the key and value matrices from previous tokens during autoregressive generation, avoiding redundant computation and significantly speeding up inference.'
  },
  {
    id: 'llm-h-7',
    question: 'What is the Chinchilla scaling law?',
    options: [
      'A method for model compression',
      'The finding that model size and training data should scale proportionally for optimal compute efficiency',
      'A technique for distributed training',
      'A new transformer architecture'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Hard',
    explanation: 'The Chinchilla paper showed that for a given compute budget, model size and training tokens should be scaled equally, suggesting many large models are undertrained and smaller models could perform better with more data.'
  },
  {
    id: 'llm-h-8',
    question: 'What is multi-query attention (MQA)?',
    options: [
      'Asking the model multiple questions at once',
      'A variant where multiple query heads share single key and value heads',
      'Running multiple attention mechanisms in parallel',
      'A training technique for better accuracy'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Hard',
    explanation: 'Multi-query attention uses multiple query heads but shares single key and value heads across all queries, reducing memory and computation during inference while maintaining performance.'
  },

  // LLMOps - Easy Questions
  {
    id: 'ops-e-1',
    question: 'What does LLMOps stand for?',
    options: [
      'Large Language Model Operations',
      'Language Learning Model Optimization',
      'Linear Learning Machine Operations',
      'Language Logic Model Output Processing'
    ],
    correctAnswer: 0,
    category: 'LLMOps',
    difficulty: 'Easy',
    explanation: 'LLMOps stands for Large Language Model Operations, encompassing the practices and tools for deploying, monitoring, and maintaining LLMs in production.'
  },
  {
    id: 'ops-e-2',
    question: 'What is the purpose of fine-tuning an LLM?',
    options: [
      'To reduce the model size',
      'To adapt the model to specific tasks or domains',
      'To speed up inference',
      'To fix bugs in the model'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Easy',
    explanation: 'Fine-tuning adapts a pre-trained model to perform better on specific tasks or domains by training it on task-specific data.'
  },
  {
    id: 'ops-e-3',
    question: 'What is inference in the context of LLMs?',
    options: [
      'Training the model',
      'Using the trained model to generate predictions',
      'Evaluating model performance',
      'Preparing training data'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Easy',
    explanation: 'Inference is the process of using a trained model to generate predictions or outputs for new inputs.'
  },
  {
    id: 'ops-e-4',
    question: 'What metric measures how long it takes for a model to generate output?',
    options: [
      'Accuracy',
      'Perplexity',
      'Latency',
      'Throughput'
    ],
    correctAnswer: 2,
    category: 'LLMOps',
    difficulty: 'Easy',
    explanation: 'Latency measures the time delay between receiving input and producing output, crucial for user experience in production systems.'
  },
  {
    id: 'ops-e-5',
    question: 'What is model quantization?',
    options: [
      'Counting model parameters',
      'Reducing numerical precision to decrease model size',
      'Measuring model quality',
      'Splitting the model across devices'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Easy',
    explanation: 'Quantization reduces the precision of model weights (e.g., from 32-bit to 8-bit), decreasing model size and inference cost with minimal accuracy loss.'
  },

  // LLMOps - Medium Questions
  {
    id: 'ops-m-1',
    question: 'What is the difference between LoRA and full fine-tuning?',
    options: [
      'LoRA is faster but less accurate',
      'LoRA only updates a small number of additional parameters',
      'LoRA requires more GPU memory',
      'LoRA cannot be used with large models'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Medium',
    explanation: 'LoRA (Low-Rank Adaptation) adds small trainable rank decomposition matrices to model layers, updating far fewer parameters than full fine-tuning while maintaining performance.'
  },
  {
    id: 'ops-m-2',
    question: 'What is the purpose of a model registry in LLMOps?',
    options: [
      'To register API keys',
      'To store and version trained models',
      'To monitor model performance',
      'To schedule training jobs'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Medium',
    explanation: 'A model registry is a centralized repository for storing, versioning, and managing trained models, facilitating model governance and deployment.'
  },
  {
    id: 'ops-m-3',
    question: 'What is A/B testing in the context of LLM deployment?',
    options: [
      'Testing two different datasets',
      'Comparing model performance between different GPUs',
      'Testing two model versions with different user groups',
      'Testing accuracy vs. bias'
    ],
    correctAnswer: 2,
    category: 'LLMOps',
    difficulty: 'Medium',
    explanation: 'A/B testing deploys different model versions to separate user groups to compare their real-world performance and make data-driven deployment decisions.'
  },
  {
    id: 'ops-m-4',
    question: 'What is the purpose of caching in LLM inference?',
    options: [
      'To reduce memory usage',
      'To store and reuse previously computed results',
      'To improve model accuracy',
      'To enable offline inference'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Medium',
    explanation: 'Caching stores responses to common queries or intermediate computations, reducing latency and computational costs for repeated or similar inputs.'
  },
  {
    id: 'ops-m-5',
    question: 'What is the main benefit of using RLHF (Reinforcement Learning from Human Feedback)?',
    options: [
      'Faster training times',
      'Aligning model outputs with human preferences',
      'Reducing model size',
      'Improving tokenization'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Medium',
    explanation: 'RLHF uses human feedback to train reward models that guide the LLM to produce outputs that better align with human values and preferences.'
  },
  {
    id: 'ops-m-6',
    question: 'What is prompt injection in LLM security?',
    options: [
      'Optimizing prompts for better results',
      'An attack where malicious instructions are embedded in user input to manipulate model behavior',
      'Adding prompts to training data',
      'A technique for faster inference'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Medium',
    explanation: 'Prompt injection is a security vulnerability where attackers craft inputs that override or manipulate the model\'s intended instructions, potentially causing harmful outputs.'
  },
  {
    id: 'ops-m-7',
    question: 'What is the purpose of a safety classifier in LLM deployments?',
    options: [
      'To classify model versions',
      'To detect and filter harmful inputs or outputs',
      'To categorize user queries',
      'To improve model accuracy'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Medium',
    explanation: 'Safety classifiers are separate models that screen inputs and outputs for harmful content, policy violations, or sensitive information before reaching users.'
  },
  {
    id: 'ops-m-8',
    question: 'What is model drift in production?',
    options: [
      'Models moving between servers',
      'Model performance degrading over time as data distributions change',
      'Weights changing during inference',
      'Models generating off-topic responses'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Medium',
    explanation: 'Model drift occurs when the distribution of production data diverges from training data over time, causing performance degradation and requiring retraining or updates.'
  },
  {
    id: 'ops-m-9',
    question: 'What is guardrails in LLM applications?',
    options: [
      'Physical server protection',
      'Rules and constraints to control model behavior and outputs',
      'API rate limiting',
      'User authentication systems'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Medium',
    explanation: 'Guardrails are programmatic controls that constrain LLM inputs/outputs to ensure safe, compliant, and on-topic responses, including content filters, fact-checking, and output validation.'
  },

  // LLMOps - Hard Questions
  {
    id: 'ops-h-1',
    question: 'In continuous batching for LLM inference, what problem does it solve?',
    options: [
      'Model accuracy degradation',
      'Inefficient GPU utilization from variable sequence lengths',
      'Memory leaks during long inference sessions',
      'Model drift over time'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Hard',
    explanation: 'Continuous batching (or iteration-level batching) allows new requests to join a batch as soon as any sequence completes, maximizing GPU utilization despite variable generation lengths.'
  },
  {
    id: 'ops-h-2',
    question: 'What is the key challenge that PagedAttention (used in vLLM) addresses?',
    options: [
      'Slow attention computation',
      'Memory fragmentation from storing KV caches',
      'Poor model accuracy',
      'High training costs'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Hard',
    explanation: 'PagedAttention manages KV cache memory in non-contiguous paged blocks, similar to virtual memory in operating systems, reducing memory fragmentation and enabling higher throughput.'
  },
  {
    id: 'ops-h-3',
    question: 'What is the trade-off when using speculative decoding?',
    options: [
      'Lower accuracy for higher speed',
      'Higher memory usage for faster inference',
      'Reduced interpretability for better performance',
      'Worse calibration for faster training'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Hard',
    explanation: 'Speculative decoding uses a smaller draft model to predict multiple tokens, which are then verified by the larger model in parallel. This trades memory for speed without sacrificing quality.'
  },
  {
    id: 'ops-h-4',
    question: 'In tensor parallelism for LLM inference, how are model parameters distributed?',
    options: [
      'Each device gets a copy of the full model',
      'Different layers are placed on different devices',
      'Model weights within layers are split across devices',
      'Only activations are distributed'
    ],
    correctAnswer: 2,
    category: 'LLMOps',
    difficulty: 'Hard',
    explanation: 'Tensor parallelism splits individual weight matrices within each layer across multiple devices, enabling models larger than single-GPU memory and reducing per-device memory.'
  },
  {
    id: 'ops-h-5',
    question: 'What is the primary purpose of Constitutional AI in LLM alignment?',
    options: [
      'To make models smaller and faster',
      'To use AI feedback based on principles instead of human feedback',
      'To improve multilingual capabilities',
      'To reduce training costs'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Hard',
    explanation: 'Constitutional AI uses a set of principles (a "constitution") to have the AI critique and revise its own outputs, reducing reliance on extensive human feedback for alignment.'
  },

  // GenAI - Easy Questions
  {
    id: 'gen-e-1',
    question: 'What does GenAI stand for?',
    options: [
      'General Artificial Intelligence',
      'Generative Artificial Intelligence',
      'Generic AI Application',
      'Genetic Algorithm Integration'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Easy',
    explanation: 'GenAI stands for Generative Artificial Intelligence, referring to AI systems that can create new content like text, images, code, or audio.'
  },
  {
    id: 'gen-e-2',
    question: 'What is a prompt in the context of LLMs?',
    options: [
      'A warning message',
      'The input text given to the model to generate a response',
      'A hyperparameter during training',
      'An error message'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Easy',
    explanation: 'A prompt is the input text or instruction given to an LLM to guide it in generating a desired response or completing a task.'
  },
  {
    id: 'gen-e-3',
    question: 'What does RAG stand for in GenAI?',
    options: [
      'Random Access Generation',
      'Retrieval-Augmented Generation',
      'Recursive Algorithm Generation',
      'Rapid AI Generation'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Easy',
    explanation: 'RAG stands for Retrieval-Augmented Generation, a technique that retrieves relevant information from external sources to enhance LLM responses.'
  },
  {
    id: 'gen-e-4',
    question: 'What are embeddings in the context of AI?',
    options: [
      'Compressed model files',
      'Vector representations of data',
      'Training checkpoints',
      'API endpoints'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Easy',
    explanation: 'Embeddings are dense vector representations that capture semantic meaning, allowing similar concepts to be close together in vector space.'
  },
  {
    id: 'gen-e-5',
    question: 'What is few-shot learning in LLMs?',
    options: [
      'Training with limited data',
      'Providing examples in the prompt to guide the model',
      'Using small models',
      'Fast inference techniques'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Easy',
    explanation: 'Few-shot learning provides a few examples in the prompt to demonstrate the desired task or format, helping the model understand what is expected without fine-tuning.'
  },

  // GenAI - Medium Questions
  {
    id: 'gen-m-1',
    question: 'What is the purpose of prompt engineering?',
    options: [
      'To train models faster',
      'To craft effective prompts that elicit desired model behavior',
      'To reduce model size',
      'To automate data collection'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Medium',
    explanation: 'Prompt engineering involves designing and refining prompts to get the best possible outputs from LLMs, including techniques like few-shot examples, chain-of-thought, and system instructions.'
  },
  {
    id: 'gen-m-2',
    question: 'What is the difference between zero-shot and few-shot prompting?',
    options: [
      'Zero-shot uses no examples, few-shot provides examples',
      'Zero-shot is faster than few-shot',
      'Few-shot requires fine-tuning',
      'Zero-shot only works with small models'
    ],
    correctAnswer: 0,
    category: 'GenAI',
    difficulty: 'Medium',
    explanation: 'Zero-shot prompting asks the model to perform a task without examples, while few-shot prompting includes examples in the prompt to demonstrate the desired behavior.'
  },
  {
    id: 'gen-m-3',
    question: 'In RAG systems, what is the purpose of the retrieval step?',
    options: [
      'To fetch the model weights',
      'To find relevant documents to provide as context',
      'To download training data',
      'To retrieve cached responses'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Medium',
    explanation: 'The retrieval step searches a knowledge base or document collection for relevant information to include in the prompt, grounding the LLM\'s response in specific sources.'
  },
  {
    id: 'gen-m-4',
    question: 'What is chain-of-thought prompting?',
    options: [
      'Linking multiple models together',
      'Asking the model to show its reasoning steps',
      'Processing prompts in sequence',
      'Training models on logic problems'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Medium',
    explanation: 'Chain-of-thought prompting encourages the model to break down complex problems and show its reasoning process, often improving accuracy on multi-step tasks.'
  },
  {
    id: 'gen-m-5',
    question: 'What is semantic search in the context of embeddings?',
    options: [
      'Searching by exact keyword match',
      'Searching based on meaning rather than exact words',
      'Searching through model architectures',
      'Searching for syntax errors'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Medium',
    explanation: 'Semantic search uses embeddings to find results based on meaning and context rather than exact keyword matches, improving relevance for natural language queries.'
  },
  {
    id: 'gen-m-6',
    question: 'What is the purpose of chunking in RAG systems?',
    options: [
      'To compress documents',
      'To split large documents into smaller segments for embedding and retrieval',
      'To organize training data',
      'To reduce API costs'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Medium',
    explanation: 'Chunking breaks documents into smaller, semantically meaningful segments that fit within embedding model limits and provide more precise retrieval results.'
  },
  {
    id: 'gen-m-7',
    question: 'What is a vector database?',
    options: [
      'A database for storing vector graphics',
      'A specialized database optimized for storing and searching high-dimensional embeddings',
      'A database with array data types',
      'A distributed database system'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Medium',
    explanation: 'Vector databases (like Pinecone, Weaviate, Chroma) are optimized for storing embeddings and performing fast similarity search using techniques like approximate nearest neighbor search.'
  },
  {
    id: 'gen-m-8',
    question: 'What is function calling (tool use) in LLMs?',
    options: [
      'Calling Python functions during training',
      'LLMs generating structured API calls to external tools based on user requests',
      'Internal function calls within the model',
      'Callback functions in the API'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Medium',
    explanation: 'Function calling allows LLMs to generate structured calls to external APIs/tools when they determine it\'s needed, enabling actions like database queries, calculations, or web searches.'
  },
  {
    id: 'gen-m-9',
    question: 'What is hallucination in LLMs?',
    options: [
      'Visual artifacts in image generation',
      'When models generate plausible-sounding but factually incorrect information',
      'Random output generation',
      'Model errors during training'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Medium',
    explanation: 'Hallucination occurs when LLMs confidently generate false information that sounds plausible, a key challenge requiring mitigation through RAG, fact-checking, and careful prompting.'
  },

  // GenAI - Hard Questions
  {
    id: 'gen-h-1',
    question: 'What is the "lost in the middle" problem in long-context LLMs?',
    options: [
      'Models forgetting training data',
      'Reduced attention to information in the middle of long contexts',
      'Errors in the middle layers of the network',
      'Performance degradation during training'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Hard',
    explanation: 'Research shows LLMs perform better on information at the beginning and end of long contexts, with reduced recall for information in the middle, affecting RAG system design.'
  },
  {
    id: 'gen-h-2',
    question: 'In vector databases for RAG, what does the HNSW algorithm optimize?',
    options: [
      'Storage space',
      'Approximate nearest neighbor search speed',
      'Embedding quality',
      'Indexing accuracy'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Hard',
    explanation: 'HNSW (Hierarchical Navigable Small World) is a graph-based algorithm that enables fast approximate nearest neighbor search in high-dimensional spaces, crucial for efficient RAG retrieval.'
  },
  {
    id: 'gen-h-3',
    question: 'What is the key advantage of using HyDE (Hypothetical Document Embeddings) in RAG?',
    options: [
      'Faster embedding computation',
      'Generating hypothetical answers to improve retrieval',
      'Reducing embedding dimensions',
      'Better compression of documents'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Hard',
    explanation: 'HyDE generates a hypothetical answer to the query and uses it for retrieval, often matching better with actual documents than the original question would.'
  },
  {
    id: 'gen-h-4',
    question: 'What problem does the ReAct (Reasoning + Acting) framework address?',
    options: [
      'Slow inference speeds',
      'Combining reasoning traces with actions for tool use',
      'Model hallucinations',
      'Memory constraints'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Hard',
    explanation: 'ReAct interleaves reasoning (thinking about what to do) with actions (calling tools/APIs), creating a synergistic approach where reasoning guides actions and observations inform reasoning.'
  },
  {
    id: 'gen-h-5',
    question: 'In prompt engineering, what is "jailbreaking"?',
    options: [
      'Removing model safety constraints through adversarial prompts',
      'Accessing restricted model APIs',
      'Bypassing rate limits',
      'Extracting model weights'
    ],
    correctAnswer: 0,
    category: 'GenAI',
    difficulty: 'Hard',
    explanation: 'Jailbreaking refers to crafting prompts that attempt to bypass a model\'s safety guidelines and content policies to generate prohibited outputs, a key security concern in LLM deployment.'
  },
  {
    id: 'gen-h-6',
    question: 'What is the self-consistency technique in chain-of-thought prompting?',
    options: [
      'Checking if the model contradicts itself',
      'Generating multiple reasoning paths and selecting the most consistent answer',
      'Ensuring prompt formatting is consistent',
      'Validating model outputs against training data'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Hard',
    explanation: 'Self-consistency generates multiple chain-of-thought reasoning paths with different random seeds, then selects the answer that appears most frequently, improving accuracy on complex reasoning tasks.'
  },
  {
    id: 'gen-h-7',
    question: 'What is the purpose of re-ranking in RAG pipelines?',
    options: [
      'Sorting database records',
      'Re-scoring retrieved documents with a more sophisticated model to improve relevance',
      'Reorganizing the knowledge base',
      'Updating document priorities'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Hard',
    explanation: 'Re-ranking uses a cross-encoder or more sophisticated model to re-score initially retrieved documents, often improving relevance by considering query-document interactions more deeply than initial embedding similarity.'
  },

  // LLM - Easy Questions (Additional)
  {
    id: 'llm-e-6',
    question: 'What does GPT stand for?',
    options: [
      'General Purpose Transformer',
      'Generative Pre-trained Transformer',
      'Global Processing Technology',
      'Gradient Propagation Training'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Easy',
    explanation: 'GPT stands for Generative Pre-trained Transformer, representing models that are pre-trained on large text corpora and can generate coherent text.'
  },
  {
    id: 'llm-e-7',
    question: 'What is an embedding layer in neural networks?',
    options: [
      'A layer that compresses images',
      'A layer that converts tokens into dense vector representations',
      'The final output layer',
      'A layer for data augmentation'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Easy',
    explanation: 'An embedding layer maps discrete tokens (words, subwords) into continuous dense vectors that capture semantic relationships and can be processed by neural networks.'
  },
  {
    id: 'llm-e-8',
    question: 'What is the purpose of the softmax function in language models?',
    options: [
      'To make training faster',
      'To convert logits into probability distributions over vocabulary',
      'To reduce model size',
      'To normalize input data'
    ],
    correctAnswer: 1,
    category: 'LLM',
    difficulty: 'Easy',
    explanation: 'Softmax converts raw model outputs (logits) into a probability distribution, allowing the model to predict the likelihood of each token in the vocabulary being the next token.'
  },

  // LLMOps - Easy Questions (Additional)
  {
    id: 'ops-e-6',
    question: 'What is batching in model inference?',
    options: [
      'Processing multiple inputs together for efficiency',
      'Splitting data into training batches',
      'Grouping API requests by user',
      'Organizing model versions'
    ],
    correctAnswer: 0,
    category: 'LLMOps',
    difficulty: 'Easy',
    explanation: 'Batching groups multiple inference requests together to process simultaneously, improving GPU utilization and throughput in production systems.'
  },
  {
    id: 'ops-e-7',
    question: 'What is the purpose of logging in production LLM systems?',
    options: [
      'To record user activity for marketing',
      'To track inputs, outputs, errors, and metrics for monitoring and debugging',
      'To save model weights',
      'To create user documentation'
    ],
    correctAnswer: 1,
    category: 'LLMOps',
    difficulty: 'Easy',
    explanation: 'Logging captures requests, responses, errors, latencies, and other metrics to enable monitoring, debugging, performance analysis, and compliance in production systems.'
  },

  // GenAI - Easy Questions (Additional)
  {
    id: 'gen-e-6',
    question: 'What is a system prompt?',
    options: [
      'A prompt for system administrators',
      'Initial instructions that set the behavior and role of the AI assistant',
      'Error messages from the system',
      'Prompts generated by the operating system'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Easy',
    explanation: 'A system prompt provides initial instructions that define the AI\'s role, personality, constraints, and behavior guidelines for the entire conversation.'
  },
  {
    id: 'gen-e-7',
    question: 'What is context in an LLM conversation?',
    options: [
      'The training data used',
      'The conversation history and relevant information available to the model',
      'The model architecture',
      'The server environment'
    ],
    correctAnswer: 1,
    category: 'GenAI',
    difficulty: 'Easy',
    explanation: 'Context includes the conversation history, system prompts, and any additional information (like retrieved documents) that the model uses to generate relevant responses.'
  },

  // Forecasting - Easy Questions
  {
    id: 'fc-e-1',
    question: 'What is time series forecasting?',
    options: [
      'Predicting future values based on historical time-ordered data',
      'Analyzing data at a specific point in time',
      'Sorting data by timestamp',
      'Converting timestamps to different time zones'
    ],
    correctAnswer: 0,
    category: 'Forecasting',
    difficulty: 'Easy',
    explanation: 'Time series forecasting uses historical observations ordered by time to predict future values, essential for demand planning, inventory management, and resource allocation.'
  },
  {
    id: 'fc-e-2',
    question: 'What does ARIMA stand for?',
    options: [
      'Automated Regression Integrated Moving Analysis',
      'AutoRegressive Integrated Moving Average',
      'Advanced Regression and Integration Model Algorithm',
      'Adaptive Real-time Integrated Modeling Approach'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Easy',
    explanation: 'ARIMA stands for AutoRegressive Integrated Moving Average, a popular statistical model for time series forecasting that combines autoregression, differencing, and moving averages.'
  },
  {
    id: 'fc-e-3',
    question: 'What is a simple moving average?',
    options: [
      'The average of all historical data points',
      'The average of a fixed window of recent observations',
      'The weighted average giving more importance to recent data',
      'The average rate of change over time'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Easy',
    explanation: 'A simple moving average calculates the arithmetic mean of a fixed number of recent observations, smoothing out short-term fluctuations to identify trends.'
  },
  {
    id: 'fc-e-4',
    question: 'What is seasonality in time series?',
    options: [
      'Random fluctuations in data',
      'Regular, predictable patterns that repeat over fixed periods',
      'The overall upward or downward trend',
      'Outliers that occur during specific seasons'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Easy',
    explanation: 'Seasonality refers to regular patterns that repeat at fixed intervals (daily, weekly, monthly, yearly), such as increased retail sales during holidays or higher electricity demand in summer.'
  },
  {
    id: 'fc-e-5',
    question: 'What is a forecast horizon?',
    options: [
      'The accuracy threshold for predictions',
      'The time period into the future for which predictions are made',
      'The historical data range used for training',
      'The confidence interval around predictions'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Easy',
    explanation: 'The forecast horizon is the length of time into the future for which forecasts are generated, such as 7-day, 30-day, or 12-month ahead forecasts.'
  },
  {
    id: 'fc-e-6',
    question: 'What is MAE in forecasting evaluation?',
    options: [
      'Maximum Absolute Error',
      'Mean Absolute Error',
      'Median Adjusted Estimation',
      'Model Accuracy Evaluation'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Easy',
    explanation: 'MAE (Mean Absolute Error) measures the average magnitude of errors between predicted and actual values, providing an easy-to-interpret metric in the same units as the data.'
  },
  {
    id: 'fc-e-7',
    question: 'What does "stationary" mean in time series analysis?',
    options: [
      'The data does not change over time',
      'Statistical properties like mean and variance are constant over time',
      'The data has no missing values',
      'The data has no seasonal patterns'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Easy',
    explanation: 'A stationary time series has constant mean, variance, and autocorrelation over time. Many forecasting models like ARIMA require stationary data to work properly.'
  },
  {
    id: 'fc-e-8',
    question: 'What is the purpose of differencing in time series?',
    options: [
      'To remove outliers from the data',
      'To remove trends and make the series stationary',
      'To fill in missing values',
      'To increase the forecast horizon'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Easy',
    explanation: 'Differencing subtracts consecutive observations to remove trends and seasonality, helping to make a non-stationary series stationary. This is the "I" (Integrated) part of ARIMA.'
  },
  {
    id: 'fc-e-9',
    question: 'Which model is best for capturing linear trends with no seasonality?',
    options: [
      'Simple Moving Average',
      'Linear Regression with time as a feature',
      'Random Forest',
      'K-Nearest Neighbors'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Easy',
    explanation: 'Linear regression with time as a feature is a simple and effective method for capturing linear trends in time series data without seasonal components.'
  },

  // Forecasting - Medium Questions
  {
    id: 'fc-m-1',
    question: 'What are the three components of an ARIMA(p,d,q) model?',
    options: [
      'p=periods, d=data points, q=quality score',
      'p=autoregressive order, d=differencing degree, q=moving average order',
      'p=prediction window, d=distribution type, q=quantile level',
      'p=polynomial degree, d=decay rate, q=quantization level'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'In ARIMA(p,d,q): p is the number of autoregressive terms, d is the degree of differencing needed to make the series stationary, and q is the number of moving average terms.'
  },
  {
    id: 'fc-m-2',
    question: 'What are the main components of Facebook Prophet model?',
    options: [
      'Linear regression with time features',
      'Trend, seasonality, holidays, and error term',
      'ARIMA with exogenous variables',
      'Neural network with attention mechanism'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'Prophet decomposes time series into trend (growth), seasonality (periodic patterns), holidays (special events), and an error term, making it intuitive and effective for business forecasting.'
  },
  {
    id: 'fc-m-3',
    question: 'Why is RMSE more sensitive to large errors than MAE?',
    options: [
      'RMSE uses a different calculation method',
      'RMSE squares the errors before averaging, penalizing large errors more',
      'RMSE is calculated on a log scale',
      'MAE ignores outliers'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'RMSE (Root Mean Squared Error) squares the errors before averaging, which disproportionately penalizes large errors. This makes RMSE higher than MAE when large errors exist.'
  },
  {
    id: 'fc-m-4',
    question: 'What is the difference between train-test split in time series vs. regular machine learning?',
    options: [
      'There is no difference',
      'Time series must maintain temporal order; no random shuffling',
      'Time series uses a larger test set',
      'Time series requires equal-sized splits'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'Time series data must preserve temporal order when splitting. Training data comes first chronologically, followed by test data, to simulate real-world forecasting where you predict the future.'
  },
  {
    id: 'fc-m-5',
    question: 'What is exponential smoothing?',
    options: [
      'Removing exponential trends from data',
      'A forecasting method that assigns exponentially decreasing weights to older observations',
      'Applying logarithmic transformation to smooth data',
      'A neural network activation function for time series'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'Exponential smoothing gives more weight to recent observations and exponentially less weight to older ones, making it responsive to recent changes while incorporating historical patterns.'
  },
  {
    id: 'fc-m-6',
    question: 'In demand planning, what is the bullwhip effect?',
    options: [
      'Sudden spikes in demand during promotions',
      'Amplification of demand variability as you move up the supply chain',
      'The lag between order and delivery',
      'Seasonal fluctuations in demand'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'The bullwhip effect describes how small fluctuations in retail demand can cause progressively larger swings in demand at wholesale, distributor, and manufacturer levels.'
  },
  {
    id: 'fc-m-7',
    question: 'What is MAPE used for in forecasting?',
    options: [
      'Mean Absolute Percentage Error - measuring forecast accuracy as a percentage',
      'Maximum Allowed Prediction Error - setting accuracy thresholds',
      'Model Architecture Performance Evaluation',
      'Multi-step Ahead Prediction Error'
    ],
    correctAnswer: 0,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'MAPE (Mean Absolute Percentage Error) expresses forecast accuracy as a percentage, making it scale-independent and easy to interpret (e.g., 5% MAPE means average 5% error).'
  },
  {
    id: 'fc-m-8',
    question: 'What are the typical steps for building an ARIMA model?',
    options: [
      'Train model → Make predictions → Evaluate',
      'Check stationarity → Identify p,d,q parameters → Fit model → Validate residuals → Forecast',
      'Clean data → Split train/test → Run model → Deploy',
      'Normalize data → Grid search → Select best model'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'ARIMA workflow: 1) Test for stationarity (ADF test), 2) Make stationary if needed (differencing), 3) Identify parameters using ACF/PACF plots, 4) Fit model, 5) Validate residuals are white noise, 6) Generate forecasts.'
  },
  {
    id: 'fc-m-9',
    question: 'What is the main difference between ARIMA and Prophet?',
    options: [
      'ARIMA is faster than Prophet',
      'ARIMA is statistical and requires stationarity; Prophet is additive decomposition and handles trends/seasonality automatically',
      'Prophet cannot handle missing data',
      'ARIMA only works with daily data'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'ARIMA is a classical statistical model requiring stationary data and manual parameter tuning. Prophet is designed for business forecasting, automatically handling trends, multiple seasonalities, and holidays without requiring stationarity.'
  },
  {
    id: 'fc-m-10',
    question: 'SCENARIO: You need to forecast daily retail sales with strong weekly seasonality and holiday effects. Which model is most appropriate?',
    options: [
      'Simple Moving Average',
      'Linear Regression',
      'Prophet or SARIMA',
      'Random Walk'
    ],
    correctAnswer: 2,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'Prophet excels at business time series with multiple seasonalities and holidays. SARIMA can also handle weekly seasonality. Both are better than simple methods that ignore seasonal patterns and special events.'
  },
  {
    id: 'fc-m-11',
    question: 'What is the difference between ACF and PACF plots?',
    options: [
      'ACF shows all correlations; PACF shows direct correlations removing indirect effects',
      'ACF is for autoregressive terms; PACF is for moving average terms',
      'There is no difference',
      'ACF is for stationary data; PACF is for non-stationary data'
    ],
    correctAnswer: 0,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'ACF (Autocorrelation Function) shows correlation with all lags. PACF (Partial Autocorrelation Function) shows direct correlation at each lag, controlling for shorter lags. Used together to identify ARIMA(p,q) parameters.'
  },
  {
    id: 'fc-m-12',
    question: 'SCENARIO: Your data has a clear upward trend but no seasonality and limited historical data (6 months). Which model would you choose?',
    options: [
      'SARIMA with seasonal components',
      'Exponential Smoothing (Holt\'s method) or simple ARIMA',
      'Complex neural network',
      'Prophet with multiple seasonalities'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'With limited data and just a trend, simpler models like Holt\'s exponential smoothing or ARIMA(p,1,q) are most appropriate. Complex models would overfit with only 6 months of data.'
  },
  {
    id: 'fc-m-13',
    question: 'What does a residual plot tell you in forecasting model validation?',
    options: [
      'The accuracy of the forecast',
      'Whether the model has captured all patterns (residuals should be random/white noise)',
      'The trend direction',
      'The seasonal component strength'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Medium',
    explanation: 'Good model residuals should look like white noise - random, normally distributed with mean zero and constant variance. Patterns in residuals indicate the model missed some structure in the data.'
  },

  // Forecasting - Hard Questions
  {
    id: 'fc-h-1',
    question: 'What do the additional parameters in SARIMA(p,d,q)(P,D,Q)m represent?',
    options: [
      'Multiple seasonal patterns at different frequencies',
      'Seasonal autoregressive, differencing, and moving average orders with period m',
      'Parallel model ensembles for improved accuracy',
      'Polynomial terms for non-linear trend modeling'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'SARIMA extends ARIMA with seasonal components: P (seasonal AR order), D (seasonal differencing), Q (seasonal MA order), and m (seasonal period, e.g., 12 for monthly data with yearly seasonality).'
  },
  {
    id: 'fc-h-2',
    question: 'In Prophet, what does the changepoint_prior_scale parameter control?',
    options: [
      'The number of seasonal components',
      'The flexibility of the trend; higher values allow more trend changes',
      'The weight given to holiday effects',
      'The smoothness of seasonality patterns'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'changepoint_prior_scale controls trend flexibility in Prophet. Higher values (e.g., 0.5) make the trend more flexible and responsive to changes, while lower values (e.g., 0.001) keep it smoother.'
  },
  {
    id: 'fc-h-3',
    question: 'What is time series cross-validation (rolling origin)?',
    options: [
      'Standard k-fold cross-validation on time series data',
      'Iteratively training on expanding windows and testing on subsequent periods',
      'Splitting data into random training and test sets',
      'Validating forecasts against multiple baseline models'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'Time series cross-validation uses expanding or rolling windows: train on [1:n], test on [n+1:n+h], then train on [1:n+k], test on [n+k+1:n+k+h], maintaining temporal order throughout.'
  },
  {
    id: 'fc-h-4',
    question: 'In Databricks for production forecasting, what is the recommended architecture pattern?',
    options: [
      'Single notebook with all forecasting logic',
      'Delta Lake for data storage, MLflow for experiment tracking, scheduled jobs for retraining',
      'CSV files with Python scripts in production',
      'Real-time streaming with Kafka for all forecasts'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'Production forecasting in Databricks typically uses Delta Lake for versioned data storage, MLflow for model tracking/registry, and scheduled jobs for automated retraining and inference pipelines.'
  },
  {
    id: 'fc-h-5',
    question: 'What is MASE (Mean Absolute Scaled Error) and why is it useful?',
    options: [
      'A metric that normalizes MAE by the in-sample naive forecast error',
      'The maximum absolute error scaled by the forecast horizon',
      'A weighted average of multiple error metrics',
      'The median absolute error standardized by variance'
    ],
    correctAnswer: 0,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'MASE scales the MAE by the MAE of a naive forecast (typically one-step ahead), making it scale-independent and interpretable across different datasets (MASE < 1 means better than naive).'
  },
  {
    id: 'fc-h-6',
    question: 'What is the Augmented Dickey-Fuller (ADF) test used for?',
    options: [
      'Testing for seasonality patterns',
      'Testing for stationarity in time series data',
      'Determining optimal ARIMA parameters',
      'Detecting outliers and anomalies'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'The ADF test is a statistical test for stationarity. It tests the null hypothesis that a unit root is present (non-stationary). Low p-values (<0.05) suggest the series is stationary.'
  },
  {
    id: 'fc-h-7',
    question: 'In forecasting hierarchical demand (total → region → store), what is reconciliation?',
    options: [
      'Combining multiple model predictions through voting',
      'Ensuring forecasts at different levels sum up correctly (e.g., stores sum to region)',
      'Adjusting forecasts based on actual sales data',
      'Removing inconsistencies in historical data'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'Hierarchical reconciliation ensures coherence: store-level forecasts sum to regional forecasts, which sum to total. Methods include bottom-up, top-down, and optimal reconciliation approaches.'
  },
  {
    id: 'fc-h-8',
    question: 'What is the key advantage of XGBoost for time series forecasting over traditional methods?',
    options: [
      'It automatically handles missing data',
      'It can capture complex non-linear relationships and interactions between features',
      'It requires less training data',
      'It provides built-in confidence intervals'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'XGBoost excels at capturing non-linear patterns and feature interactions (e.g., how day-of-week and promotions interact). It requires feature engineering to create lag features, rolling statistics, etc.'
  },
  {
    id: 'fc-h-9',
    question: 'What is intermittent demand forecasting and why is it challenging?',
    options: [
      'Forecasting demand that occurs sporadically with many zero values',
      'Predicting demand during business interruptions',
      'Short-term forecasting with limited data',
      'Forecasting seasonal products with gaps in availability'
    ],
    correctAnswer: 0,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'Intermittent demand has many zero values (e.g., spare parts, slow-moving items). Traditional methods struggle with zeros; specialized methods like Croston\'s method or bootstrapping are often needed.'
  },
  {
    id: 'fc-h-10',
    question: 'In Databricks Auto ML for forecasting, what does it automatically handle?',
    options: [
      'Only data loading and visualization',
      'Feature engineering, model selection, hyperparameter tuning, and MLflow tracking',
      'Just model training without evaluation',
      'Data collection from external sources'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'Databricks AutoML automates the entire pipeline: creates lag features, time-based features, tests multiple algorithms (Prophet, ARIMA, XGBoost), tunes hyperparameters, and logs everything to MLflow.'
  },
  {
    id: 'fc-h-11',
    question: 'SCENARIO: You have 5 years of hourly electricity demand data with daily and yearly seasonality, weather impacts, and holiday effects. Which approach is best?',
    options: [
      'Single ARIMA model',
      'Prophet with multiple seasonalities and exogenous weather variables, or XGBoost with engineered features',
      'Simple exponential smoothing',
      'Naive forecast with seasonal adjustment'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'This scenario needs multiple seasonalities, external regressors (weather), and holiday handling. Prophet handles this well, or XGBoost with proper feature engineering (hour, day-of-week, month, weather features, lag features).'
  },
  {
    id: 'fc-h-12',
    question: 'What is the Box-Jenkins methodology for ARIMA modeling?',
    options: [
      'A grid search approach for hyperparameters',
      'An iterative process: identification (ACF/PACF) → estimation (fitting) → diagnostic checking (residuals)',
      'A neural network training procedure',
      'A method for handling missing data'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'The Box-Jenkins methodology is the systematic approach to ARIMA: 1) Identify p,d,q using ACF/PACF plots and stationarity tests, 2) Estimate parameters by fitting the model, 3) Check residuals for white noise, iterate if needed.'
  },
  {
    id: 'fc-h-13',
    question: 'When comparing ARIMA vs. Exponential Smoothing, which statement is TRUE?',
    options: [
      'They are completely different and incompatible',
      'Every exponential smoothing model has an equivalent ARIMA representation',
      'ARIMA is always more accurate',
      'Exponential smoothing cannot handle trends'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'There is a mathematical equivalence between exponential smoothing state space models and certain ARIMA models. For example, Simple Exponential Smoothing ≡ ARIMA(0,1,1), and Holt\'s method ≡ ARIMA(0,2,2).'
  },
  {
    id: 'fc-h-14',
    question: 'SCENARIO: You need to forecast demand for 10,000 SKUs daily. What is the most scalable approach?',
    options: [
      'Train individual ARIMA models for each SKU manually',
      'Use automated forecasting frameworks (Prophet, AutoARIMA) with parallel processing in Databricks/Spark',
      'Use a single global model for all SKUs',
      'Use simple moving averages for all SKUs'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'For large-scale forecasting, use automated frameworks that can be parallelized. Databricks with Spark can train thousands of models in parallel. Libraries like AutoARIMA or Prophet with pandas UDFs in Spark are ideal for this scenario.'
  },
  {
    id: 'fc-h-15',
    question: 'What is the purpose of the Ljung-Box test in ARIMA modeling?',
    options: [
      'To test for stationarity',
      'To test if residuals are independently distributed (white noise)',
      'To determine the optimal p,d,q parameters',
      'To compare different models'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'The Ljung-Box test checks if residuals exhibit autocorrelation. A non-significant p-value (>0.05) suggests residuals are white noise, indicating the model has captured all temporal patterns.'
  },
  {
    id: 'fc-h-16',
    question: 'SCENARIO: Historical sales show a structural break (new competitor entered market 6 months ago). How should you handle this?',
    options: [
      'Use all historical data equally',
      'Use only post-break data, or use models that can detect changepoints (Prophet), or add a regressor for the intervention',
      'Ignore the break and use simple averages',
      'Remove the break period from training data'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'Structural breaks require special handling: 1) Train only on recent data post-break, 2) Use models with changepoint detection like Prophet, 3) Include intervention variables, or 4) Use ARIMAX with a step function for the break.'
  },
  {
    id: 'fc-h-17',
    question: 'When would you choose XGBoost over Prophet for time series forecasting?',
    options: [
      'When you have many exogenous variables and non-linear relationships',
      'When you have limited computational resources',
      'When you need automatic seasonality detection',
      'XGBoost is never appropriate for time series'
    ],
    correctAnswer: 0,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'XGBoost excels when you have many external variables (promotions, weather, economics) with complex non-linear interactions. It requires manual feature engineering (lags, rolling stats) but can capture intricate patterns Prophet might miss.'
  },
  {
    id: 'fc-h-18',
    question: 'What is forecast reconciliation in demand planning hierarchies?',
    options: [
      'Adjusting forecasts based on domain expert input',
      'Ensuring forecasts at different aggregation levels are coherent and sum correctly',
      'Comparing multiple model forecasts',
      'Correcting forecast errors after actuals arrive'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'Forecast reconciliation ensures coherence across hierarchy levels (product → category → total). Methods include bottom-up (sum SKUs), top-down (allocate total), middle-out, and optimal reconciliation using MinT or other approaches.'
  },
  {
    id: 'fc-h-19',
    question: 'SCENARIO: Your forecast intervals are too wide to be useful for planning. What should you investigate?',
    options: [
      'Use a different model immediately',
      'Check for high variance in data, model misspecification, or consider ensemble methods to reduce uncertainty',
      'Ignore uncertainty and use point forecasts',
      'Reduce the forecast horizon'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'Wide intervals indicate high uncertainty. Investigate: 1) Data quality/variance, 2) Model fit (residual diagnostics), 3) Feature engineering to explain more variance, 4) Ensemble methods to reduce uncertainty, 5) Whether data is inherently noisy.'
  },
  {
    id: 'fc-h-20',
    question: 'In a Databricks MLflow production pipeline, what is the recommended model deployment pattern for forecasting?',
    options: [
      'Retrain models manually when performance degrades',
      'Scheduled retraining with model registry, staged rollout, and automated monitoring for drift',
      'Deploy once and never update',
      'Retrain daily regardless of performance'
    ],
    correctAnswer: 1,
    category: 'Forecasting',
    difficulty: 'Hard',
    explanation: 'Best practice: 1) Schedule regular retraining (weekly/monthly), 2) Register models in MLflow, 3) Use staging (Dev→Staging→Production), 4) Monitor for concept drift and data quality, 5) Automate rollback if metrics degrade, 6) Version control everything.'
  }
];

// Utility function to get questions by filters
export function getQuestionsByFilter({ category, difficulty, limit }) {
  let filtered = questions;

  if (category) {
    filtered = filtered.filter(q => q.category === category);
  }

  if (difficulty) {
    filtered = filtered.filter(q => q.difficulty === difficulty);
  }

  if (limit) {
    filtered = filtered.slice(0, limit);
  }

  return filtered;
}

// Get random questions with balanced distribution
export function getRandomQuestions(count = 30, selectedCategories = ['LLM', 'LLMOps', 'GenAI', 'Forecasting'], difficultyDistribution = null) {
  // Filter questions by selected categories
  const filteredQuestions = questions.filter(q => selectedCategories.includes(q.category));

  // Shuffle function
  const shuffle = (array) => {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  };

  let selectedEasy, selectedMedium, selectedHard;

  if (difficultyDistribution) {
    // Use custom distribution
    const easy = filteredQuestions.filter(q => q.difficulty === 'Easy');
    const medium = filteredQuestions.filter(q => q.difficulty === 'Medium');
    const hard = filteredQuestions.filter(q => q.difficulty === 'Hard');

    selectedEasy = shuffle(easy).slice(0, difficultyDistribution.easy);
    selectedMedium = shuffle(medium).slice(0, difficultyDistribution.medium);
    selectedHard = shuffle(hard).slice(0, difficultyDistribution.hard);
  } else {
    // Default balanced distribution
    const questionsPerDifficulty = Math.floor(count / 3);
    const easy = filteredQuestions.filter(q => q.difficulty === 'Easy');
    const medium = filteredQuestions.filter(q => q.difficulty === 'Medium');
    const hard = filteredQuestions.filter(q => q.difficulty === 'Hard');

    selectedEasy = shuffle(easy).slice(0, questionsPerDifficulty);
    selectedMedium = shuffle(medium).slice(0, questionsPerDifficulty);
    selectedHard = shuffle(hard).slice(0, questionsPerDifficulty);
  }

  // Combine and shuffle all selected questions
  const combined = [...selectedEasy, ...selectedMedium, ...selectedHard];

  // If we don't have enough questions, fill with whatever is available
  if (combined.length < count) {
    const remaining = shuffle(filteredQuestions.filter(q => !combined.includes(q)));
    combined.push(...remaining.slice(0, count - combined.length));
  }

  return shuffle(combined);
}
