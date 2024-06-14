<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>
<body>

<h1>NER Project for Movie Descriptions</h1>

<p>Welcome to the Named Entity Recognition (NER) project for detecting actors, plots, awards, and more from movie descriptions using the RoBERTa transformer model.</p>

<h2>Introduction</h2>
<p>This project leverages advanced natural language processing (NLP) techniques to analyze movie descriptions and extract key entities. By utilizing the RoBERTa transformer model, we can achieve high accuracy in identifying crucial details such as:</p>
<ul>
    <li>Actors</li>
    <li>Character Names</li>
    <li>Directors</li>
    <li>Genres</li>
    <li>Plot Elements</li>
    <li>Years</li>
    <li>Soundtracks</li>
    <li>Opinions</li>
    <li>Awards</li>
    <li>Origins</li>
    <li>Quotes</li>
    <li>Relationships</li>
</ul>

<h2>Architecture</h2>
<p>The backbone of this project is built on the RoBERTa (Robustly optimized BERT approach) transformer model:</p>
<ul>
    <li><strong>RoBERTa:</strong> An optimized version of BERT, designed to improve performance on a variety of NLP tasks by using larger mini-batches and training on more data.</li>
</ul>

<h2>Use Cases</h2>
<p>This NER project can be applied in various scenarios, including but not limited to:</p>
<ul>
    <li><strong>Information Extraction:</strong> Automatically extract detailed information from movie descriptions, reviews, and summaries.</li>
    <li><strong>Content Summarization:</strong> Generate concise summaries and metadata for movies, aiding in quick comprehension.</li>
    <li><strong>Recommendation Systems:</strong> Enhance recommendation algorithms by providing structured data from movie descriptions.</li>
    <li><strong>Content Management:</strong> Improve the organization and retrieval of movie-related data.</li>
</ul>

<h2>Installation</h2>
<p>To get started with this project, follow these steps:</p>
<ol>
    <li>Clone the repository: <code>git clone https://github.com/yourusername/ner-movie-descriptions.git</code></li>
    <li>Navigate to the project directory: <code>cd ner-movie-descriptions</code></li>
    <li>Install the required dependencies: <code>pip install -r requirements.txt</code></li>
</ol>

<h2>Usage</h2>
<p>To run the NER model on a movie description, use the following steps:</p>
<ol>
    <li>Load the model architecture.</li>
    <li>Load the pre-trained model weights using <code>torch.load_state_dict</code>. The weights are provided as a state dictionary.</li>
    <li>Model weights: https://drive.google.com/file/d/1-0GjiiF9CfGImp7V2sVViKudz7KXOPb4/view?usp=share_link</li>

</ol>

<p>Model Architecture:</p>
<pre><code>
from transformers import RobertaModel, RobertaTokenizerFast
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
model_name = "roberta-base"
base_model = RobertaModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CustomRoBERTaForTokenClassification(nn.Module):
    def __init__(self,num_labels):
        super(CustomRoBERTaForTokenClassification, self).__init__()
        self.roberta = base_model
        self.dropout = nn.Dropout(0.1)
        #classifier
        self.classifier = nn.Linear(768,num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        #passing through classifier
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits

num_labels = 25
model = CustomRoBERTaForTokenClassification(num_labels).to(device)</code></pre>

<h2>Contributing</h2>
<p>We welcome contributions to enhance this project. Please fork the repository and submit pull requests.</p>

<h2>License</h2>
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>



