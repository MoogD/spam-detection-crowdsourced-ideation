# spam-detection-crowdsourced-ideation

# Requirements
Will be updated soon

# Usage
To use the Spam detection execute the spamdetection.py file with a path to a csv or xml file with ideas:
``` 
python spamdetection.py "path" [-h] [-t, --train] [--challenge] ["Challenge"]
```

Use the option `-t` or `--train` to train the system.
Else the System will classify the ideas.

The option `--challenge`can be used to give a challenge to the ideas.
If no challenge is given, the spamdetection will check for a challenge in an idea ("CHALLENGE" field in csv files) and if there is no challenge it will use a domain independent trained model for the training/classification.

# Input
You can use the System to classify ideas saved in xml or csv files. 

**CSV:** Ideas in a csv file just need to have an "DESCRIPTION" field that contains the content of the idea. E.g: 
```
"ID","DESCRIPTION","STATUS","CREATED","WORKER_ID"
```
    
**XML:** Ideas in an xml file need to be in a `gi2mo:Idea` tag with `gi2mo="http://purl.org/gi2mo/ns#"`The idea tag need to have a `gi2mo:content` attribute that contains the content of the idea. E.g: 
```
<gi2mo:Idea ...>
     ...
     <gi2mo:content>Content of the Idea</gi2mo:content>
</gi2mo:Idea>
```

If you want to use the trainingsoption (currently just working with csv files) the ideas need an additional fiel for the actual classification. This field can be called "STATUS" or "SPAM" ("STATUS" will be prioritized if both fields are available).
The value of the "STATUS" field needs to be "unusable" for spam-ideas or usable for ham-ideas.
The value of the "SPAM" field needs to be "spam" or "ham". Other values in these fields can cause false classifications.
 
# Output
If you use the trainingsoption, the system will train models for the given challenge that will automatically be used if you classify ideas from the same challenge. 

If you use the system for classification, it will create a new file at the same path as the given file.
The new file will contain the same content plus additional fields/attributes for the results of the classification.

**CSV:** For csv files there will be three more fields added: "DUPLICATE" (containing "yes" or "no"), "SPAMPROB" (value between 0.00 and 1.00) and "TRIGGERED" (containing a list of Filters that indicate that the idea is spam)

**XML:** For xml files there will be two more attributes added: "Duplicat" (containing if the idea is a duplicat or not) and "Spamsystem" (containing the spamprobability and the triggered filters). E.g:
```
<Duplicate Duplicate="no" />
<Spamsystem Spamprob="0.81" Triggered="['1-WordBayes: 1.0', '5-WordBayes: 1.0', 'SentenceEmbedding: 0.5918028950691223', 'linearClassifier: 0.6550258427083513', 'containsName']" />
```


