import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setting uri scheme -> agar bhool gaye ho tho session 15 lecture ke timestamp 1:21:00 pe jaao.
mlflow.set_tracking_uri('https://dagshub.com/Rahul-Data-Scientist/session_16_part_1.mlflow')

import dagshub
dagshub.init(repo_owner='Rahul-Data-Scientist', repo_name='session_16_part_1', mlflow=True)

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model
max_depth = 100

mlflow.set_experiment('iris-dt')

# Start an MLflow run
with mlflow.start_run():
    # there is a parameter in mlflow.start_run() by the name 'run_name' which is used to set the run name. For example - run_name = 'your desired name'
    dt = DecisionTreeClassifier(max_depth = max_depth)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # creating heatmap plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names)
    
    # save the plot as an artifact
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    # logging the code -> Yeh hum git/github se karte hain mainly, but ab mlflow seekh rahe hain toh yeh bhi karna seekh lete hain.
    mlflow.log_artifact(__file__)

    # logging the model
    mlflow.sklearn.log_model(dt, artifact_path = "Decision Tree")
    # Waise hum yeh bhi kar sakte the - mlflow.log_model(dt, 'Decision Tree')
    # But, since yeh model sklearn se aaya hai, so mlflow.sklearn karne se mlflow iss model ka aur useful metadata leke aata hai as you can see in the ui.

    # setting tags - although yeh hum mlflow ui mein bhi kar sakte hain as run name pe click karne se jo naya page khulta hai, usme add tag ka option hota hai. BUt code se bhi karna seekh lete hain.
    mlflow.set_tag('author', 'rahul')
    mlflow.set_tag('model', 'decision tree')
    # tags use karne se hum apne run ko easily search kar sakte hain using the search bar in mlflow ui.


    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)

    print('accuracy score', accuracy)