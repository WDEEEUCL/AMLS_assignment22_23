# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import numpy as np
from tensorflow.keras.preprocessing import image
import os



basedir = 'C:\\Users\\Wei Dai\\PycharmProjects\\AMLS_22-23_SN19111862\\Datasets\\dataset_AMLS_22-23\\cartoon_set'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

def extract_features_labels():
    print('Start extracting features and labels')
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    #target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()

    face_shape_labels = {line.split('\t')[-1].split('.')[-2] : int(line.split('\t')[2]) for line in lines[1:]} # face shape (0-4)

    width = 30
    height = 30
    dim = (width, height)

    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name= img_path.split('.')[-2].split('\\')[-1]
            print('Extracting image: ', file_name)

            # load image
            img = image.img_to_array(image.load_img(img_path, target_size=dim, interpolation='bicubic'))
            #features, _ = run_dlib_shape(img)
            #if features is not None:
            all_features.append(img)
            all_labels.append(face_shape_labels[file_name])

    img = np.array(all_features)
    face_shape_labels = np.array(all_labels) # simply converts the -1 into 0, so male=0 and female=1
    #return img, face_shape_labels
    print('Extraction complete')
    return img, face_shape_labels


###Load CelebA data and create train and test splits (Train: 100 exmaples, Test: 100 examples)
def get_data():
    # X, y = import_data.extract_features_labels()
    X, y = extract_features_labels()

    # totoal number of train data
    tn_train = len(y)
    Y = np.zeros((tn_train, 5))
    for i in range(tn_train):
        Y[i, int(y[i])] = 1

    tr_X = X[:7500] ; tr_Y = Y[:7500]
    te_X = X[7500:] ; te_Y = Y[7500:]

    return tr_X, tr_Y, te_X, te_Y

#X, y = extract_features_labels()
#tr_X, tr_Y, te_X, te_Y = get_data()
#print(X.shape)
#print(tr_X.shape)

###Allocate memory for weights and biases for all MLP layers
def allocate_weights_and_biases():
    print('Allocating memory for weights and biases')
    # define number of hidden layers ..
    n_hidden_1 = 40  # 1st layer number of neurons
    n_hidden_2 = 40  # 2nd layer number of neurons

    n_outC = 5 # number of output classes
    # inputs placeholders
    #X = tf.placeholder("float", [None, 68, 2])
    X = tf.placeholder("float", [None, 30, 30, 3])
    Y = tf.placeholder("float", [None, n_outC])  # number of output classes


    # flatten image features into one vector (i.e. reshape image feature matrix into a vector)
    # images_flat = tf.contrib.layers.flatten(X)
    images_flat = tf.compat.v1.layers.flatten(X)

    # weights and biases are initialized from a normal distribution with a specified standard devation stddev
    stddev = 0.01 #Standard deviation

    # define placeholders for weights and biases in the graph
    weights = {
        #'hidden_layer1': tf.Variable(tf.random_normal([68 * 2, n_hidden_1], stddev=stddev)),
        'hidden_layer1': tf.Variable(tf.random_normal([30 * 30 * 3, n_hidden_1], stddev=stddev)),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_outC], stddev=stddev))
    }

    biases = {
        'bias_layer1': tf.Variable(tf.random_normal([n_hidden_1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([n_outC], stddev=stddev))
    }
    print('Memory allocation complete')
    return weights, biases, X, Y, images_flat

#weights, biases, X, Y, images_flat = allocate_weights_and_biases()


###Define how the weights and biases are used for inferring classes from inputs (i.e. define MLP function)
# Create model
def multilayer_perceptron():
    print('Creating the MLP model')
    weights, biases, X, Y, images_flat = allocate_weights_and_biases()

    # Hidden fully connected layer 1
    layer_1 = tf.add(tf.matmul(images_flat, weights['hidden_layer1']), biases['bias_layer1'])
    #layer_1 = tf.sigmoid(layer_1)
    layer_1 = tf.keras.activations.relu(layer_1)

    # Hidden fully connected layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['bias_layer2'])
    #layer_2 = tf.sigmoid(layer_2)
    layer_2 = tf.keras.activations.relu(layer_2)

    # Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    print('Model creation complete')
    return out_layer, X, Y

#out_layer, X, Y = multilayer_perceptron()

#Define graph training operation
# learning parameters
print('Defining the training operation for graph')
learning_rate = 1e-5
training_epochs = 10000

# display training accuracy every ..
display_accuracy_step = 10

training_images, training_labels, test_images, test_labels = get_data()
logits, X, Y = multilayer_perceptron()

# define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# define training graph operation
train_op = optimizer.minimize(loss_op)

# graph operation to initialize all variables
init_op = tf.global_variables_initializer()

print('Graph construction complete')


###Run graph for specified number of epochs.
with tf.Session() as sess:
    # run graph weights/biases initialization op
    sess.run(init_op)
    # begin training loop ..
    print('training begin')
    for epoch in range(training_epochs):
        # complete code below
        # run optimization operation (backprop) and cost operation (to get loss value)
        _, cost = sess.run([train_op, loss_op], feed_dict={X: training_images, Y: training_labels})

        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(cost))

        if epoch % display_accuracy_step == 0:
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

            # calculate training accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy: {:.3f}".format(accuracy.eval({X: training_images, Y: training_labels})))

    print("Optimization Finished!")

    # -- Define and run test operation -- #

    # apply softmax to output logits
    pred = tf.nn.softmax(logits)

    #  derive inffered calasses as the class with the top value in the output density function
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # complete code below
    # run test accuracy operation ..

    print("Test Accuracy:", accuracy.eval({X: test_images, Y: test_labels}))