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
