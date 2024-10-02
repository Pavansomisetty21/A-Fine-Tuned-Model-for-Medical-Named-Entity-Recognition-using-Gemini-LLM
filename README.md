# A Fine Tuned Model For Medical NER Using Gemini LLM
<div align="center">
  
![Screenshot 2024-10-01 144634](https://github.com/user-attachments/assets/9ae17542-cc32-4971-b3c9-5910ab0a9057)

<b>Fine-Tuning of LLM</b>
</div>

To our NER Fine-tuning model we got  output for below input as 
```python
input="she was suffering from Malaria and fits so to cure  she used amodiaquine and she has slightly fever and pain on hip"

### For the above input our tuned model got output as

Entity: Malaria, Label: DISEASE
Entity: fits, Label: SYMPTOM
Entity: amodiaquine, Label: DRUG
Entity: fever, Label: SYMPTOM
Entity: pain, Label: SYMPTOM
Entity: hip, Label: BODY_PART
```
In the above example input it extract all medical entity terms and give names for it 

Now Quick start a description on NER and Fine-Tuning and after Explanation of code

**Named Entity Recognition (NER)** is a subtask of information extraction and natural language processing (NLP) that focuses on identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, dates, quantities, monetary values, and other entities. NER is crucial for understanding unstructured data, as it transforms textual content into structured information, enabling downstream tasks like information retrieval, summarization, and question answering.


### **History of Named Entity Recognition:**

1. **Early Beginnings (1990s)**:
   - The first significant developments in NER began in the early 1990s, emerging from the information extraction (IE) community. The motivation was driven by the need to extract structured information from unstructured text in fields like news articles and legal documents.
   - The **Message Understanding Conferences (MUC)** series, initiated by DARPA (Defense Advanced Research Projects Agency) in the early 1990s, played a crucial role in fostering research in this area. The 6th MUC in 1995 was pivotal for NER, where it became an independent task for identifying entities such as names, dates, locations, etc.

2. **Rule-based and Statistical Approaches (1990s - Early 2000s)**:
   - Early NER systems were predominantly **rule-based**, relying on handcrafted rules and lexicons. These systems were highly domain-specific and required manual effort to adapt to different languages or domains.
   - **Statistical models** like Hidden Markov Models (HMMs), Decision Trees, and Maximum Entropy models emerged in the late 1990s and early 2000s. These approaches leveraged machine learning and probabilistic methods to improve generalization and performance, but they still required feature engineering and large labeled datasets.

3. **Rise of Machine Learning (2000s)**:
   - In the 2000s, **Conditional Random Fields (CRFs)** became popular for NER. CRFs are a type of discriminative model used for sequence labeling tasks. This approach outperformed earlier methods by capturing the dependencies between neighboring words or entities in the sequence.
   - **Support Vector Machines (SVMs)** and other supervised learning techniques also contributed to advancements in NER, reducing the reliance on handcrafted features by learning patterns from labeled datasets.

4. **Neural Network and Deep Learning Era (2010s - Present)**:
   - By the 2010s, **deep learning** revolutionized NER, with approaches such as **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** models becoming popular. These models capture long-range dependencies in text and automate feature extraction, leading to significant improvements in NER tasks.
   - The introduction of **word embeddings** like **Word2Vec** and **GloVe** enabled NER systems to learn semantic relationships between words. Combining embeddings with RNNs and LSTMs improved accuracy in various languages and domains.
   - **Bidirectional LSTMs (BiLSTMs)** combined with **CRFs** became the standard for many NER tasks in the mid-2010s.
  
5. **Transformer Models and Pre-trained Language Models (Late 2010s - Present)**:
   - The development of transformer-based models, like **BERT (Bidirectional Encoder Representations from Transformers)**, revolutionized NER. These pre-trained models capture context better than previous architectures and can be fine-tuned for specific NER tasks.
   - **BERT-based NER models** consistently achieve state-of-the-art performance on many benchmarks due to their ability to understand the context of words in both directions.
   - More recently, models such as **RoBERTa**, **GPT**, **T5**, and **Gemini** have also been employed for NER tasks, offering even greater flexibility in multi-task learning and domain adaptation.

### **Current Trends**:
- **Multilingual NER**: With global applications of NLP, there is a growing focus on multilingual NER, leveraging models like **mBERT** and **XLM-R** that can recognize entities across multiple languages.
- **Domain-Specific NER**: NER has been applied in specialized domains like biomedical NER (for extracting drug names, diseases) and legal NER (for identifying legal terms).
- **Few-shot Learning and Zero-shot Learning**: Recent trends in machine learning are focusing on NER models that require less data or can adapt to new entity types with minimal or no training data.

NER remains a core component of many NLP applications, driving advancements in information extraction and knowledge representation.

----

In this repo we use current trending Named entity Recognition **Domain-Specific NER** ,In this repo we fine-tune the Gemini model for **MEDICAL Named Entity Recognition**


**Fine-tuning** is the process of taking a pre-trained model and making minor adjustments to adapt it to a specific task or dataset. In deep learning and natural language processing (NLP), it allows leveraging large, pre-trained models (such as BERT, GPT, or T5) and specializing them for a particular downstream task with less computational effort and a smaller dataset.

### **Key Concepts of Fine-tuning**:

1. **Pre-training and Fine-tuning**:
   - In the pre-training phase, models are trained on large-scale, general-purpose datasets (e.g., Wikipedia, books, web data) using unsupervised or self-supervised learning. This helps the model learn general language patterns, word representations, and knowledge about the world.
   - Fine-tuning involves taking this pre-trained model and continuing its training on a smaller, task-specific dataset. During fine-tuning, the model's parameters are updated to learn the nuances of the specific task, such as sentiment analysis, named entity recognition (NER), or text classification.

2. **Advantages of Fine-tuning**:
   - **Reduced Training Time**: Since the model is already pre-trained, it requires significantly less time to converge on the new task.
   - **Smaller Dataset**: Fine-tuning can work with relatively smaller datasets compared to training a model from scratch.
   - **Better Performance**: Pre-trained models capture a vast amount of linguistic and world knowledge. Fine-tuning allows the model to retain this knowledge while adapting to the nuances of a specific task, often leading to superior performance compared to training a model from scratch.

3. **Transfer Learning**:
   Fine-tuning is an example of **transfer learning**, where knowledge learned from one domain (during pre-training) is transferred to another domain (during fine-tuning). This approach is particularly effective in NLP because many tasks share common linguistic patterns.

### **Steps in Fine-tuning**:

1. **Select a Pre-trained Model**:
   Choose a pre-trained model that aligns with the task. Popular choices for NLP include:
   - **BERT** for sentence classification, NER, and question answering.
   - **GPT** for text generation tasks.
   - **T5** for sequence-to-sequence tasks like summarization, translation, and text-to-text generation.
   - **GEMINI** for NLP,Multi-model development

2. **Prepare Task-Specific Data**:
   The next step is to collect and format a dataset specific to the task at hand.
   For example:
   - For sentiment analysis, label sentences as positive, negative, or neutral.
   - For NER, label entities like names, dates, and locations within text.

     In this we take **Medical Entities** like symptom,Disease,Body part ... etc
     
3. **Fine-tuning Process**:
   - **Add a Task-Specific Layer**: Depending on the model architecture, a simple classifier (like a softmax layer) is usually added on top of the pre-trained model to generate task-specific outputs.
   - **Freeze or Unfreeze Layers**: In some cases, you might freeze the early layers of the pre-trained model and only fine-tune the last few layers. Alternatively, you can fine-tune all layers of the model.
   - **Training**: Train the model on your task-specific data. During this phase, the learning rate is usually set lower than when training from scratch, to avoid "catastrophic forgetting" of the knowledge learned during pre-training.

4. **Evaluate the Fine-tuned Model**:
   After fine-tuning, evaluate the model on a validation or test set to check its performance on the specific task. Fine-tuned models are often tested for accuracy, precision, recall, and F1 scores.

### **Challenges in Fine-tuning**:
- **Overfitting**: Fine-tuning with a very small dataset can lead to overfitting, where the model performs well on training data but poorly on unseen data.
- **Catastrophic Forgetting**: If the learning rate is too high or the fine-tuning is too aggressive, the model may forget the general knowledge it acquired during pre-training.
- **Resource Requirements**: Though fine-tuning is faster than training from scratch, it still requires substantial computational resources, especially for large models like GPT-3 or T5.

### **Fine-tuning in Different Domains**:

1. **Natural Language Processing (NLP)**:
   Fine-tuning is widely used in NLP for tasks such as text classification, named entity recognition, machine translation, text summarization, and question answering. Pre-trained models like BERT, GPT, RoBERTa, T5, and others are adapted to these specific tasks through fine-tuning.

2. **Computer Vision**:
   Fine-tuning is also common in computer vision, where pre-trained models like ResNet, EfficientNet, or VGG, trained on large image datasets like ImageNet, are adapted to domain-specific tasks such as medical image classification or object detection.

3. **Speech and Audio Processing**:
   Fine-tuning can be applied to pre-trained audio models to adapt them for tasks like speech recognition, speaker identification, or emotion detection in voice.


### **Popular Tools and Frameworks** for Fine-tuning:
- **Hugging Face Transformers**: Hugging Face provides an easy-to-use interface for fine-tuning pre-trained language models like BERT, GPT, T5, and others.
- **TensorFlow**: TensorFlow has tools for fine-tuning models using its pre-trained models or custom architectures.
- **PyTorch**: PyTorch is a popular framework for deep learning and supports fine-tuning with a high level of customization.
- **Fast.ai**: Fast.ai provides high-level abstractions for fine-tuning models with minimal effort.
- **Google AI Studio**: It is a website/environment where we can tune gemini models with our own dataset

### **Applications of Fine-tuning**:
- **Sentiment Analysis**: Fine-tuning pre-trained models on domain-specific customer reviews to classify sentiments.
- **Named Entity Recognition (NER)**: Fine-tuning models like BERT on annotated datasets for identifying entities such as people, organizations, and locations.
- **Machine Translation**: Adapting general translation models to specific language pairs or dialects.
- **Text Summarization**: Fine-tuning sequence-to-sequence models (e.g., T5) on domain-specific data to generate summaries for articles or reports.

Fine-tuning makes it possible to tailor powerful pre-trained models for specific tasks efficiently, thereby leveraging the strengths of both general and task-specific learning.

----


# Steps Involved in Fine-Tuning

This project utilizes the Google Generative AI SDK to analyze medical statements. It can provide insights into the accuracy and clarity of the statements related to medical conditions, treatments, and symptoms.


## Installation

To install the Google Generative AI SDK, run the following command:

```bash
pip install google-generativeai
```

## Usage

Follow these steps to set up and use the Google Generative AI SDK for medical Named Entity Recognition :

1. **Import the Required Library:**
   This step imports the Google Generative AI library, which allows us to access the API and its functionalities.
   ```python
   import google.generativeai as genai
   ```

2. **Configure the API Key:**
   Set your API key to authenticate your requests. Replace `"your_api_key_here"` with your actual API key.
   
   ```python
   apikey = "your_api_key_here"
   genai.configure(api_key=apikey)
   ```

3. **List Available Tuned Models:**
   This step lists the available tuned models you can use for generating responses. It helps you identify which models are accessible.
   
   ```python
   for i, m in zip(range(5), genai.list_tuned_models()):
       print(m.name)
   ```

4. **Select a Base Model:**
   Here, we select the first base model that supports the creation of tuned models. This model will serve as the foundation for our generative model.
   
   ```python
   base_model = [
       m for m in genai.list_models()
       if "createTunedModel" in m.supported_generation_methods][0]
   ```

5. **Create a Generative Model Configuration:**
   
   This configuration sets parameters like temperature, top_p, and output token limits that control the generation behavior of the model.
   
   ```python
   generation_config = {
       "temperature": 0.2,  # Controls the randomness of the output
       "top_p": 0.95,  # Controls diversity via nucleus sampling
       "top_k": 64,  # Limits the number of highest probability tokens to consider
       "max_output_tokens": 8192,  # Maximum number of tokens to generate in the response
       "response_mime_type": "text/plain",  # Format of the output
   }
   ```

6. **Initialize the Generative Model:**
    
   Create an instance of the generative model using your tuned model's name. This allows you to generate responses based on the model's training.
   
   <div align="center">
     <b>In this step we use our own finetuned model which was tuned with own well written dataset</b>
   </div>
   
   ```python
   
   model = genai.GenerativeModel(model_name="tunedModels/your_tuned_model_name_here",  # Replace with your tuned model my model was (medical-jmf8sizpikmcfg)
       generation_config=generation_config,
   )
  
   ```

7. **Start a Chat Session with Some History:**
    
   Initialize a chat session by providing some context from previous interactions. This helps the model understand the ongoing conversation.
   ```python
   chat_session = model.start_chat(
       history=[
           {"role": "user", "parts": ["he was suffering from cancer"]},
           {"role": "model", "parts": ["The statement is correct and understandable. ..."]},  # Add model response
           {"role": "user", "parts": ["she used clobazam 10 mg for fits"]},
           {"role": "model", "parts": ["The statement is potentially problematic. ..."]},  # Add model response
       ]
   )
   ```

8. **Send a Message for Analysis:**
    
   You can now send a new medical statement for analysis. The model will generate a response based on the input.
   ```python
   response = chat_session.send_message("she was suffering from Malaria and fits so to cure she used amodiaquine and she has slightly fever and pain on hip")
   ```

9. **Print the Response:**
   Finally, print the model's response to see the analysis and insights it provides regarding the medical statement.
   ```python
   print(response.text)
   ```

### My output:

For my model for above input I got output as:

```
Entity: Malaria, Label: DISEASE
Entity: fits, Label: SYMPTOM
Entity: amodiaquine, Label: DRUG
Entity: fever, Label: SYMPTOM
Entity: pain, Label: SYMPTOM
Entity: hip, Label: BODY_PART
```




Feel free to adjust any part of the text to better fit your project's needs!
