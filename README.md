### ISYS30221: Artificial Intelligence
You are expected to submit a “lab book”, that is a collection of design documents,
software, and test results, related to the design of a chatbot. The chatbot must be modular
and demonstrate the use of several major AI techniques, as well as their integration into
one user friendly system.
The lab book is to be submitted on NOW (electronic submission only) in several stages
over the academic year as detailed in the following table.
Stage Target
completion date
Tasks
1 Chatbot design planning document
Chatbot with rule-based and similarity-based component
Conversation log
2 Chatbot extended with image classification component
Conversation log
3 Chatbot extended with toy world reasoning system
Conversation log
4 Chatbot extended with sequence to sequence NN and
reinforcement learning demonstration
Conversation log
Final chatbot design specification and reflective document
Make sure that the chatbot topic, as well as the individual functionalities, are very specific,
so it’s unlikely anyone else in the class would create exactly the same functionality. You
can use other Python packages if they are on pypi.org and you find them useful for your
chosen domain of application, but you must not use them as a replacement for any of the
specified parts of the assignment (for example, you must use AIML, not any other method
to program chatbot responses as a replacement for AIML). Your code must not violate the
NTU’s Computer Use Policy, NTU’s Internet Content and Access Policy, and other relevant
laws or university policies (see
https://www4.ntu.ac.uk/information_systems/policies/index.html).
II. Assessment Scenario/Problem
Chatbot design planning document / Final chatbot design specification
This should specify the chosen topic area for the chat bot, a more detailed list of
requirements, a brief description of the individual modules you intend to create, and a flow
diagram showing how these work together to produce an answer to the user’s input. Limit
this document to two to three pages. For the final specification, update your planning
document. The final specification should also a reflective document (about one page),
which analyses what went well or not so well, puts the work in context, and mentions
some ideas for what could be done if there was more time.
Chatbot with rule-based and similarity-based component
ISYS30221: Artificial Intelligence
Coursework Specification
3
This submission should consist of one Python file that implements the chat bot, one AIML
file that implements the rules, and one file in a suitable format (such as plain text or CSV)
that has the question-answer pairs. If you wish, you can create more files than that. You
can use the Python and aiml files provided on NOW as a starting point, but you should
extend and customise them towards your design specification. The entire program might
be as compact as 110 lines but can be double that size depending on your programming
style and how many features you implement. The similarity-based component should be
based on the bag-of-words model, tf/idf, and cosine similarity. The AIML file could be as
compact as 80 lines but will probably somewhat longer, maybe even significantly longer
depending on the domain you intend your chatbot to be for.
Conversation log
For each conversation log, record an actual conversation between you (and/or your peers)
and the chatbot that demonstrates the implemented features. Annotate it with comments
that explain which feature / component generated this, and how, for any particularly
remarkable output. If the conversation included non-textual input (e.g., images), this
should also be included to the degree possible. The conversation logs will not receive a
grade by themselves, but are there to demonstrate the respective chatbot components, so
their quality influences the component grade.
Image classification component
If the user inputs a question such as “What is img01.jpg”?, the chatbot should invoke a
pre-trained convolutional neural network to provide the answer. You decide yourself which
network architecture and training data set to use. The possible classes of objects that the
network can recognise depend on the domain of the chatbot that you have chosen. If you
wish, you can decide to use a different source of images instead of files, e.g., camera
input, and also decide to trigger the network in other ways. You will need to train your
neural network using a data set with class labels. There are web sites that serve as a
repository for many good data sets such as Kaggle.com or the UCI Machine Learning
Repository. If you find it hard to find a fitting data set, you might have to change the
function specification of this component --- don’t worry too much if it doesn’t fit that well
with the remainder of the chatbot, the priority is to get a working system without spending
too much time on selecting a data set. For creating the convolutional neural network, you
must use the keras library. For loading and pre-processing the data, scikit-learn is
recommended. It is also recommended that you follow closely one of the tutorials available
on the keras or tensorflow websites (with proper referencing). One key challenge is to
make it so that the trained model can be saved and later used for classification when the
chatbot is running.
Toy world reasoning system
Use the provided NLTK based code that translates natural language sentences into firstorder logic, and does some reasoning based on its knowledge. Adapt it to your chosen
domain by adding new objects and relations (and possibly removing old ones). Connect it
to the chatbot such that it is activated when certain key words or phrases appear, adds
knowledge from user input to its knowledge base, and does some reasoning, for example,
as response to a user question. Note that this is only a toy world related system, in other
words, the number of objects and relations can remain small, there should be at least as
many as in the example provided, but not normally more than twice those numbers.
Sequence to sequence network extension
Build a sequence to sequence network, and use it in some way to generate chatbot
responses (for example, if the other modules cannot produce detect any pattern in the
input. Follow the provided tutorials to build the network, and train it with one of the data
sets of conversations that you will be pointed to. Make sure that the trained network is
saved, and loaded again when the chatbot runs. Note that as there is only a limited set of 
ISYS30221: Artificial Intelligence
Coursework Specification
4
conversation data sets available for training, some overlap with other students’ work is
unavoidable, but you must still make your own decisions about the architecture and
parameters of the neural network and training procedure.
Reinforcement learning extension
Use a reinforcement learning algorithm of your choice (either based on deep learning, or
on evolutionary algorithms) to train an agent that then becomes part of your chatbot.
Depending on the chatbot you are building, this agent could be playing a game via console
or a separate window (if you want to use OpenAI Gym), or solve some optimisation
problem (this might be most suitable if you use an evolutionary algorithm). It is again
recommended that you follow a tutorial (with proper referencing), and/or use the
examples provided in this module as a starting point. Again, as there is only a limited set
of OpenAI environments available for training, some overlap with other students’ work
might be unavoidable, but you must still make your own decisions about the architecture
and parameters of the neural network and training procedure.
