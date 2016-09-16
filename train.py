import data_utils
import numpy as np
from sklearn import cross_validation
from memn2n_kv import MemN2N_KV
from skll.metrics import kappa
import tensorflow as tf
from memn2n_kv import add_gradient_noise
import time
import os

print 'start to load flags\n'

# flags
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.2, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 50, "Batch size for training.")
tf.flags.DEFINE_integer("feature_size", 30, "Feature size")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 50, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("essay_set_id", 1, "essay set id, 1 <= id <= 8")
tf.flags.DEFINE_string("reader", "bow", "Reader for the model (bow, simple_gru)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# print flags info
# todo: save to a file for furture reference
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# hyper-parameters
essay_set_id = FLAGS.essay_set_id
batch_size = FLAGS.batch_size
embedding_size = FLAGS.embedding_size
feature_size = FLAGS.feature_size
l2_lambda = FLAGS.l2_lambda
hops = FLAGS.hops
reader = 'bow'
epochs = FLAGS.epochs
num_samples = 3

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir,
                                       "runs", timestamp))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print("Writing to {}\n".format(out_dir))
with open(out_dir+'/params', 'w') as f:
    for attr, value in sorted(FLAGS.__flags.items()):
        f.write("{}={}".format(attr.upper(), value))
        f.write("\n")

# hyper-parameters end here
training_path = 'training_set_rel3.tsv'
essay_list, resolved_scores = data_utils.load_training_data(training_path, essay_set_id)

max_score = max(resolved_scores)
min_score = min(resolved_scores)
print 'max_score is {} \t min_score is {}\n'.format(max_score, min_score)
with open(out_dir+'/params', 'a') as f:
    f.write('max_score is {} \t min_score is {} \n'.format(max_score, min_score))

# include max score
score_range = range(min_score, max_score+1)

# load glove
word_idx, word2vec = data_utils.load_glove(embedding_size)

vocab_size = len(word_idx) + 1
# stat info on data set

sent_size_list = map(len, [essay for essay in essay_list])
max_sent_size = max(sent_size_list)
mean_sent_size = int(np.mean(map(len, [essay for essay in essay_list])))

print 'max sentence size: {} \nmean sentence size: {}\n'.format(max_sent_size, mean_sent_size)
with open(out_dir+'/params', 'a') as f:
    f.write('max sentence size: {} \nmean sentence size: {}\n'.format(max_sent_size, mean_sent_size))

print 'The length of score range is {}'.format(len(score_range))
E = data_utils.vectorize_data(essay_list, word_idx, max_sent_size)

memory = []
memory_score = []
memory_sent_size = []
# pick sampled essay for each score
for i in score_range:
    for j in range(num_samples):
        if i in resolved_scores:
            score_idx = resolved_scores.index(i)
            score = resolved_scores.pop(score_idx)
            essay = E.pop(score_idx)
            sent_size = sent_size_list.pop(score_idx)
            memory.append(essay)
            memory_score.append(score)
            memory_sent_size.append(sent_size)

memory_size = len(memory)
labeled_data = zip(E, resolved_scores, sent_size_list)

# split the data on the fly
trainE, testE, train_scores, test_scores, train_sent_sizes, test_sent_sizes = cross_validation.train_test_split(E, resolved_scores, sent_size_list, test_size=.2)

trainE, evalE, train_scores, eval_scores, train_sent_sizes, eval_sent_sizes = cross_validation.train_test_split(trainE, train_scores, train_sent_sizes, test_size=.1)

# convert score to one hot encoding
train_scores_encoding = map(lambda x: score_range.index(x), train_scores)

# data size
n_train = len(trainE)
n_test = len(testE)
n_eval = len(evalE)

print 'The size of training data: {}'.format(n_train)
print 'The size of testing data: {}'.format(n_test)
print 'The size of evaluation data: {}'.format(n_eval)
with open(out_dir+'/params', 'a') as f:
    f.write('The size of training data: {}\n'.format(n_train))
    f.write('The size of testing data: {}'.format(n_test))
    f.write('The size of evaluation data: {}'.format(n_eval))

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    # decay learning rate
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.96, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)

    with tf.Session() as sess:
        model = MemN2N_KV(batch_size, vocab_size, max_sent_size, max_sent_size, memory_size,
                          memory_size, embedding_size, len(score_range), feature_size, hops, reader, l2_lambda)

        grads_and_vars = optimizer.compute_gradients(model.loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                          for g, v in grads_and_vars if g is not None]
        grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
        
        train_op = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=global_step)
        
        sess.run(tf.initialize_all_variables())

        def train_step(m, e, s):
            feed_dict = {
                model._memory_value: m,
                model._query: e,
                model._memory_key: m,
                model._score_encoding: s,
                model.keep_prob: FLAGS.keep_prob,
                model.w_placeholder: word2vec
            }
            _, step, predict_op, cost = sess.run([train_op, global_step, model.predict_op, model.cost], feed_dict)
            return predict_op, cost

        def test_step(e, m):
            feed_dict = {
                model._query: e,
                model._memory_key: m,
                model._memory_value: m,
                model.keep_prob: 1,
                model.w_placeholder: word2vec
            }
            preds = sess.run(model.predict_op, feed_dict)
            return preds

        for i in range(1, epochs+1):
            train_cost = 0
            np.random.shuffle(batches)
            for start, end in batches:
                e = trainE[start:end]
                s = train_scores_encoding[start:end]
                batched_memory = []
                # batch sized memory
                for _ in range(len(e)):
                    batched_memory.append(memory)
                _, cost = train_step(batched_memory, e, s)
                train_cost += cost
            print 'Finish epoch {}, total training cost is {}'.format(i, train_cost)
            # evaluation
            if i % FLAGS.evaluation_interval == 0 or i == FLAGS.epochs:
                # test on training data
                train_preds = []
                for start in range(0, n_train, batch_size):
                    end = min(n_train, start+batch_size)
                    
                    batched_memory = []
                    for _ in range(end-start):
                        batched_memory.append(memory)
                    preds = test_step(trainE[start:end], batched_memory)
                    for ite in preds:
                        train_preds.append(ite)
                train_preds = np.add(train_preds, min_score)
                train_kappp_score = kappa(train_scores, train_preds, 'quadratic')
            
                # test on eval data
                eval_preds = []
                for start in range(0, n_eval, batch_size):
                    end = min(n_eval, start+batch_size)
                    
                    batched_memory = []
                    for _ in range(end-start):
                        batched_memory.append(memory)
                    preds = test_step(evalE[start:end], batched_memory)
                    for ite in preds:
                        eval_preds.append(ite)

                eval_preds = np.add(eval_preds, min_score)
                eval_kappp_score = kappa(eval_scores, eval_preds, 'quadratic')

                # test on test data
                test_preds = []
                for start in range(0, n_test, batch_size):
                    end = min(n_test, start+batch_size)
                    
                    batched_memory = []
                    for _ in range(end-start):
                        batched_memory.append(memory)
                    preds = test_step(testE[start:end], batched_memory)
                    for ite in preds:
                        test_preds.append(ite)

                test_preds = np.add(test_preds, min_score)
                test_kappp_score = kappa(test_scores, test_preds, 'quadratic')
                
                print("Training kappa score = {}".format(train_kappp_score))
                print("Validation kappa score = {}".format(eval_kappp_score))
                print("Testing kappa score = {}".format(test_kappp_score))
                with open(out_dir+'/eval', 'w') as f:
                    f.write("Training kappa score = {}\n".format(train_kappp_score))
                    f.write("Validation kappa score = {}\n".format(eval_kappp_score))
                    f.write("Testing kappa score = {}\n".format(test_kappp_score))
                    f.write('*'*10)
                    f.write('\n')
