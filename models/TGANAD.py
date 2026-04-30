# ============================================================
# TGAN-AD Implementation
# (Transformer-based GAN for Anomaly Detection)
# Based on Xu et al. (2022)
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle = pos * angle_rates

        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.constant(pe)

    def call(self, x):
        return x + self.pe[: tf.shape(x)[1]]


class TransformerEncoder(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ff = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model)]
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(dropout)
        self.do2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        att = self.attn(x, x)
        att = self.do1(att, training=training)
        out1 = self.ln1(x + att)
        ff = self.ff(out1)
        ff = self.do2(ff, training=training)
        return self.ln2(out1 + ff)


class TGANAD_Generator(Model):
    """Generator: Transformer-based architecture to reconstruct normal patterns."""
    def __init__(self, past_len, n_features, d_model=64, num_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()
        self.dense_in = layers.Dense(d_model)
        self.pos_enc = PositionalEncoding(past_len, d_model)
        self.trans_enc = TransformerEncoder(d_model, num_heads, ff_dim, dropout)
        self.dense_out = layers.Dense(n_features)
        
    def call(self, x, training=False):
        x = self.dense_in(x)
        x = self.pos_enc(x)
        x = self.trans_enc(x, training=training)
        return self.dense_out(x)

class TGANAD_Discriminator(Model):
    """Discriminator: Transformer-based architecture to distinguish real vs. reconstructed."""
    def __init__(self, past_len, n_features, d_model=64, num_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()
        self.dense_in = layers.Dense(d_model)
        self.pos_enc = PositionalEncoding(past_len, d_model)
        self.trans_enc = TransformerEncoder(d_model, num_heads, ff_dim, dropout)
        self.flatten = layers.Flatten()
        self.dense_out = layers.Dense(1) # Logit for real/fake

    def call(self, x, training=False):
        x = self.dense_in(x)
        x = self.pos_enc(x)
        x = self.trans_enc(x, training=training)
        x = self.flatten(x)
        return self.dense_out(x)

class TGANAD:
    """Wrapper for TGAN-AD training loop and decision scoring."""
    def __init__(self, past_len, n_features, d_model=64, num_heads=4, ff_dim=128, dropout=0.1, lambda_adv=0.1):
        self.past_len = past_len
        self.n_features = n_features
        self.lambda_adv = lambda_adv # Weight for adversarial vs reconstruction loss
        
        self.generator = TGANAD_Generator(past_len, n_features, d_model, num_heads, ff_dim, dropout)
        self.discriminator = TGANAD_Discriminator(past_len, n_features, d_model, num_heads, ff_dim, dropout)
        
        self.g_optimizer = optimizers.Adam(1e-4)
        self.d_optimizer = optimizers.Adam(1e-4)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.threshold = None

    @tf.function
    def train_step(self, x):
        x = tf.cast(x, tf.float32)

        # 1. Train Discriminator
        with tf.GradientTape() as d_tape:
            fake_x = self.generator(x, training=True)
            real_output = self.discriminator(x, training=True)
            fake_output = self.discriminator(fake_x, training=True)
            
            d_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake
            
        grads_d = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))
        
        # 2. Train Generator
        with tf.GradientTape() as g_tape:
            fake_x = self.generator(x, training=True)
            fake_output = self.discriminator(fake_x, training=True)
            
            g_loss_adv = self.loss_fn(tf.ones_like(fake_output), fake_output)
            g_loss_rec = tf.reduce_mean(tf.abs(x - fake_x)) # L1 reconstruction loss
            g_loss = g_loss_rec + self.lambda_adv * g_loss_adv
            
        grads_g = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables))
        return d_loss, g_loss

    def fit(self, X, epochs=40, batch_size=32, verbose=0):
        X = np.asarray(X, dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(1000).batch(batch_size)
        for epoch in range(epochs):
            for batch in dataset:
                self.train_step(batch)
        
        # Calibrate threshold on the 95th percentile of normal scores
        scores = self.decision_function(X)
        self.threshold = np.quantile(scores, 0.95)
        return self

    def decision_function(self, X):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        fake_x = self.generator(X, training=False)
        rec_loss = np.mean(np.abs(X - fake_x), axis=(1, 2))
        
        # Combined Score: High reconstruction error + Low discriminator confidence
        disc_score = tf.nn.sigmoid(self.discriminator(X, training=False)).numpy().flatten()
        return rec_loss + self.lambda_adv * (1.0 - disc_score)

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)
