import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np


class MFModel:
    """
    Matrix factorisation model using TensorFlow
    r_ui = < U_u, V_i >
    """

    def __init__(self, num_users, num_items, rank, reg, min_score=None, max_score=None):
        self.rank = rank
        self.num_users = num_users
        self.num_items = num_items
        self.min_score = min_score
        self.max_score = max_score
        self.reg = reg
        self.embeddings = {
            "U": tf.Variable(tf.random_normal([self.num_users, self.rank], stddev=0.01, mean=0, seed=1), name="U"),
            "V": tf.Variable(tf.random_normal([self.num_items, self.rank], stddev=0.01, mean=0, seed=1), name="V"),
        }
        self.loss = {
            "tot": None,
            "err": None,
            "reg": None,
        }
        self.metrics = {
            "iterations": [],
            "err_loss": [],
            "reg_loss": [],
            "loss": [],
            "train_mse": [],
            "perf_mse": [],
        }

    def predict(self, users, items, clip_min=None, clip_max=None):
        U_ = tf.squeeze(tf.nn.embedding_lookup(self.embeddings["U"], users))
        V_ = tf.squeeze(tf.nn.embedding_lookup(self.embeddings["V"], items))
        predictions = tf.multiply(U_, V_)
        if self.rank > 1:
            predictions = tf.reduce_sum(predictions, reduction_indices=[1])        #predictions = tf.nn.sigmoid(predictions))
        if clip_min is not None and clip_max is not None:
            predictions = tf.clip_by_value(predictions, clip_min, clip_max)
        return predictions 
    
    def predict_final(self, users, items, clip_min=None, clip_max=None):
        predictions = np.zeros(len(users))
        i = 0
        for user, book in zip(users, items):
            u = self.embeddings["U"][user]
            v = self.embeddings["V"][book]
            pred = u.dot(v)
            if clip_min is not None and clip_max is not None:
                pred = pred.clip(clip_min, clip_max)
            predictions[i] = pred
            i += 1
        return predictions
    
    def predict_final_matrix(self, clip_min=None, clip_max=None):
        U = self.embeddings["U"]
        V = self.embeddings["V"]
        predictions = U.dot(V.T)
        if clip_min is not None and clip_max is not None:
            return predictions.clip(clip_min, clip_max)
        return predictions

    def compute_reg_loss(self):
        reg_loss = 0
        for embedding_name, embedding in self.embeddings.items():
            reg_loss += tf.reduce_mean(tf.square(embedding))  
        return reg_loss * self.reg

    def compute_loss(self, users_items_ratings):
        users, items, ratings = users_items_ratings
        prediction = self.predict(users, items)
        err_loss = tf.reduce_mean(tf.squared_difference(prediction, ratings)) 
        reg_loss = self.compute_reg_loss()
        total_loss = err_loss + reg_loss
        tf.summary.scalar("loss", total_loss)
        self.loss["err"] = err_loss
        self.loss["reg"] = reg_loss
        self.loss["tot"] = total_loss
        return total_loss

    def compute_mse(self, user_items_ratings):
        users, items, ratings = user_items_ratings
        predictions = self.predict(users, items, self.min_score, self.max_score)
        return tf.reduce_mean(tf.squared_difference(predictions, ratings))

    def train(self, train_users_items_ratings, test_users_items_ratings=None, n_iter=100, lr=0.01, plot_results=True, print_every=20):
        #self.embeddings["mu"] = tf.Variable(tf.random_normal([1], stddev=0.01, mean=train_users_items_ratings[-1].mean()), name="mu")
        tf.set_random_seed(1)
        cost = self.compute_loss(train_users_items_ratings)
        optimiser = tf.train.AdamOptimizer(lr).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for i in range(1, n_iter+1):
                sess.run(optimiser)
            
                if i == 1 or i % print_every == 0:
                    self.metrics["iterations"].append(i)
                    self.metrics["train_mse"].append(self.compute_mse(train_users_items_ratings).eval())
                    self.metrics["perf_mse"].append(self.compute_mse(test_users_items_ratings).eval())
                    self.metrics["loss"].append(self.loss["tot"].eval())
                    self.metrics["err_loss"].append(self.loss["err"].eval())
                    self.metrics["reg_loss"].append(self.loss["reg"].eval())
                    metric = {k: v[int(i/print_every)] for k, v in self.metrics.items()}
                    print("Iteration {}, {}".format(i, metric))
            
            for name, embedding in self.embeddings.items():
                self.embeddings[name] = embedding.eval()
                
            if plot_results:
                from matplotlib import pyplot as plt
                fig = plt.figure()
                fig.set_size_inches(30, 8)
                ax = fig.add_subplot(1, 1, 1)
                for metric in ["train_mse", "perf_mse"]:
                    ax.plot(self.metrics["iterations"], self.metrics[metric], label=metric)
                ax.set_xlim([1, n_iter])
                ax.legend()
                plt.xlabel("# iterations", fontsize=14)
                plt.ylabel("MSE", fontsize=14)
                plt.show()

