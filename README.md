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

CSV: Ideas in a csv file just need to have an "DESCRIPTION" field that contains the content of the idea.
     E.g: 
     ```
     "ID","DESCRIPTION","STATUS","CREATED","WORKER_ID"
     ```
    
XML: Ideas in an xml file need to be in a `gi2mo:Idea` tag with `gi2mo="http://purl.org/gi2mo/ns#"`
     The idea tag need to have a `gi2mo:content` attribute that contains the content of the idea.
     E.g: 
     ```
     <gi2mo:Idea ...>
          ...
          <gi2mo:content>Content of the Idea</gi2mo:content>
     </gi2mo:Idea>
     ```

# Output
