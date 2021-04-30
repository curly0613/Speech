#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import csv

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'

import datetime
import pickle
import shutil
import six
import subprocess
import tensorflow as tf
import time
import traceback
import inspect
import wave

from six.moves import zip, range, filter, urllib, BaseHTTPServer
from tensorflow.python.tools import freeze_graph
from threading import Thread, Lock
from util.audio import audiofile_to_input_vector
from util.feeding import DataSet, ModelFeeder
from util.gpu import get_available_gpus
from util.shared_lib import check_cupti
from util.text import sparse_tensor_value_to_texts, wer, levenshtein, Alphabet, ndarray_to_text, cer
from xdg import BaseDirectory as xdg
import numpy as np

import pyaudio
import wave

from array import array
import keyboard
import socket

# Importer
# ========
tf.app.flags.DEFINE_string  ('train_files',      '',          'comma separated list of files specifying the dataset used for training. multiple files will get merged')
tf.app.flags.DEFINE_string  ('dev_files',        '',          'comma separated list of files specifying the dataset used for validation. multiple files will get merged')
tf.app.flags.DEFINE_string  ('test_files',       '',          'comma separated list of files specifying the dataset used for testing. multiple files will get merged')
tf.app.flags.DEFINE_boolean ('fulltrace',        False,       'if full trace debug info should be generated during training')

# Cluster configuration
# =====================
tf.app.flags.DEFINE_string  ('ps_hosts',         '',          'parameter servers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('worker_hosts',     '',          'workers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('job_name',         'localhost', 'job name - one of localhost (default), worker, ps')
tf.app.flags.DEFINE_integer ('task_index',       0,           'index of task within the job - worker with index 0 will be the chief')
tf.app.flags.DEFINE_integer ('replicas',         -1,          'total number of replicas - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('replicas_to_agg',  -1,          'number of replicas to aggregate - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer  ('coord_retries',    100,         'number of tries of workers connecting to training coordinator before failing')
tf.app.flags.DEFINE_string  ('coord_host',       'localhost', 'coordination server host')
tf.app.flags.DEFINE_integer ('coord_port',       2500,        'coordination server port')
tf.app.flags.DEFINE_integer ('iters_per_worker', 1,           'number of train or inference iterations per worker before results are sent back to coordinator')

# Global Constants
# ================
tf.app.flags.DEFINE_boolean ('train',            True,        'wether to train the network')
tf.app.flags.DEFINE_boolean ('test',             True,        'wether to test the network')
tf.app.flags.DEFINE_integer ('epoch',            75,          'target epoch to train - if negative, the absolute number of additional epochs will be trained')

tf.app.flags.DEFINE_boolean ('use_warpctc',      False,       'wether to use GPU bound Warp-CTC')

tf.app.flags.DEFINE_float   ('dropout_rate',     0.05,        'dropout rate for feedforward layers')
tf.app.flags.DEFINE_float   ('dropout_rate2',    -1.0,        'dropout rate for layer 2 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate3',    -1.0,        'dropout rate for layer 3 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate4',    0.0,         'dropout rate for layer 4 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate5',    0.0,         'dropout rate for layer 5 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate6',    -1.0,        'dropout rate for layer 6 - defaults to dropout_rate')

tf.app.flags.DEFINE_float   ('relu_clip',        20.0,        'ReLU clipping value for non-recurrant layers')

# Adam optimizer (http://arxiv.org/abs/1412.6980) parameters
tf.app.flags.DEFINE_float   ('beta1',            0.9,         'beta 1 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('beta2',            0.999,       'beta 2 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('epsilon',          1e-8,        'epsilon parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('learning_rate',    0.001,       'learning rate of Adam optimizer')

# Batch sizes
tf.app.flags.DEFINE_integer ('train_batch_size', 1,           'number of elements in a training batch')
tf.app.flags.DEFINE_integer ('dev_batch_size',   1,           'number of elements in a validation batch')
tf.app.flags.DEFINE_integer ('test_batch_size',  1,           'number of elements in a test batch')

# Sample limits
tf.app.flags.DEFINE_integer ('limit_train',      0,           'maximum number of elements to use from train set - 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_dev',        0,           'maximum number of elements to use from validation set- 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_test',       0,           'maximum number of elements to use from test set- 0 means no limit')

# Step widths
tf.app.flags.DEFINE_integer ('display_step',     0,           'number of epochs we cycle through before displaying detailed progress - 0 means no progress display')
tf.app.flags.DEFINE_integer ('validation_step',  0,           'number of epochs we cycle through before validating the model - a detailed progress report is dependent on "--display_step" - 0 means no validation steps')

# Checkpointing
tf.app.flags.DEFINE_string  ('checkpoint_dir',   '',          'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('checkpoint_secs',  600,         'checkpoint saving interval in seconds')
tf.app.flags.DEFINE_integer ('max_to_keep',      5,           'number of checkpoint files to keep - default value is 5')

# Exporting
tf.app.flags.DEFINE_string  ('export_dir',       '',          'directory in which exported models are stored - if omitted, the model won\'t get exported')
tf.app.flags.DEFINE_integer ('export_version',   1,           'version number of the exported model')
tf.app.flags.DEFINE_boolean ('remove_export',    True,       'wether to remove old exported models')
tf.app.flags.DEFINE_boolean ('use_seq_length',   True,        'have sequence_length in the exported graph (will make tfcompile unhappy)')

# Reporting
tf.app.flags.DEFINE_integer ('log_level',        1,           'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
tf.app.flags.DEFINE_boolean ('log_traffic',      False,       'log cluster transaction and traffic information during debug logging')

tf.app.flags.DEFINE_string  ('wer_log_pattern',  '',          'pattern for machine readable global logging of WER progress; has to contain %%s, %%s and %%f for the set name, the date and the float respectively; example: "GLOBAL LOG: logwer(\'12ade231\', %%s, %%s, %%f)" would result in some entry like "GLOBAL LOG: logwer(\'12ade231\', \'train\', \'2017-05-18T03:09:48-0700\', 0.05)"; if omitted (default), there will be no logging')

tf.app.flags.DEFINE_boolean ('log_placement',    False,       'wether to log device placement of the operators to the console')
tf.app.flags.DEFINE_integer ('report_count',     10,          'number of phrases with lowest WER (best matching) to print out during a WER report')

tf.app.flags.DEFINE_string  ('summary_dir',      '',          'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('summary_secs',     3600,           'interval in seconds for saving TensorBoard summaries - if 0, no summaries will be written')

# Geometry
tf.app.flags.DEFINE_integer ('n_hidden',         2048,        'layer width to use when initialising layers')

# Initialization
tf.app.flags.DEFINE_integer ('random_seed',      4567,        'default random seed that is used to initialize variables')
tf.app.flags.DEFINE_float   ('default_stddev',   0.046875,    'default standard deviation to use when initialising weights and biases')

# Early Stopping
tf.app.flags.DEFINE_boolean ('early_stop',       True,        'enable early stopping mechanism over validation dataset. Make sure that dev FLAG is enabled for this to work')

# This parameter is irrespective of the time taken by single epoch to complete and checkpoint saving intervals.
# It is possible that early stopping is triggered far after the best checkpoint is already replaced by checkpoint saving interval mechanism.
# One has to align the parameters (earlystop_nsteps, checkpoint_secs) accordingly as per the time taken by an epoch on different datasets.

tf.app.flags.DEFINE_integer ('earlystop_nsteps',  4,          'number of steps to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
tf.app.flags.DEFINE_float   ('estop_mean_thresh', 0.5,        'mean threshold for loss to determine the condition if early stopping is required')
tf.app.flags.DEFINE_float   ('estop_std_thresh',  0.5,        'standard deviation threshold for loss to determine the condition if early stopping is required')

# Decoder
tf.app.flags.DEFINE_string  ('decoder_library_path', 'native_client/libctc_decoder_with_kenlm.so', 'path to the libctc_decoder_with_kenlm.so library containing the decoder implementation.')
tf.app.flags.DEFINE_string  ('alphabet_config_path', 'data/hunmin.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
tf.app.flags.DEFINE_string  ('lm_binary_path',       '', 'path to the language model binary file created with KenLM')
tf.app.flags.DEFINE_string  ('lm_trie_path',         '', 'path to the language model trie file created with native_client/generate_trie')
tf.app.flags.DEFINE_integer ('beam_width',        1024,       'beam width used in the CTC decoder when building candidate transcriptions')
tf.app.flags.DEFINE_float   ('lm_weight',         1.75,       'the alpha hyperparameter of the CTC decoder. Language Model weight.')
tf.app.flags.DEFINE_float   ('word_count_weight', 1.00,      'the beta hyperparameter of the CTC decoder. Word insertion weight (penalty).')
tf.app.flags.DEFINE_float   ('valid_word_count_weight', 1.00, 'valid word insertion weight. This is used to lessen the word insertion penalty when the inserted word is part of the vocabulary.')

# Inference mode
tf.app.flags.DEFINE_string  ('one_shot_infer',       '',       'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it. Disables training, testing and exporting.')

# Initialize from frozen model
tf.app.flags.DEFINE_string  ('initialize_from_frozen_model', '', 'path to frozen model to initialize from. This behaves like a checkpoint, loading the weights from the frozen model and starting training with those weights. The optimizer parameters aren\'t restored, so remember to adjust the learning rate.')

for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
    tf.app.flags.DEFINE_float('%s_stddev' % var, None, 'standard deviation to use when initialising %s' % var)

FLAGS = tf.app.flags.FLAGS

# Logging functions
# =================
def prefix_print(prefix, message):
    print(prefix + ('\n' + prefix).join(message.split('\n')))

def log_debug(message):
    if FLAGS.log_level == 0:
        prefix_print('D ', message)

def log_traffic(message):
    if FLAGS.log_traffic:
        log_debug(message)

def log_info(message):
    if FLAGS.log_level <= 1:
        prefix_print('I ', message)

def log_warn(message):
    if FLAGS.log_level <= 2:
        prefix_print('W ', message)

def log_error(message):
    if FLAGS.log_level <= 3:
        prefix_print('E ', message)

class TrainingCoordinator(object):
    class TrainingCoordinationHandler(BaseHTTPServer.BaseHTTPRequestHandler):
        '''Handles HTTP requests from remote workers to the Training Coordinator.
        '''
        def _send_answer(self, data=None):
            self.send_response(200)
            self.send_header('content-type', 'text/plain')
            self.end_headers()
            if data:
                self.wfile.write(data)

        def do_GET(self):
            if COORD.started:
                if self.path.startswith(PREFIX_NEXT_INDEX):
                    index = COORD.get_next_index(self.path[len(PREFIX_NEXT_INDEX):])
                    if index >= 0:
                        self._send_answer(str(index).encode("utf-8"))
                        return
                elif self.path.startswith(PREFIX_GET_JOB):
                    job = COORD.get_job(worker=int(self.path[len(PREFIX_GET_JOB):]))
                    if job:
                        self._send_answer(pickle.dumps(job))
                        return
                self.send_response(204) # end of training
            else:
                self.send_response(202) # not ready yet
            self.end_headers()

        def do_POST(self):
            if COORD.started:
                src = self.rfile.read(int(self.headers['content-length']))
                job = COORD.next_job(pickle.loads(src))
                if job:
                    self._send_answer(pickle.dumps(job))
                    return
                self.send_response(204) # end of training
            else:
                self.send_response(202) # not ready yet
            self.end_headers()

        def log_message(self, format, *args):
            '''Overriding base method to suppress web handler messages on stdout.
            '''
            return


    def __init__(self):
        ''' Central training coordination class.
        Used for distributing jobs among workers of a cluster.
        Instantiated on all workers, calls of non-chief workers will transparently
        HTTP-forwarded to the chief worker instance.
        '''
        self._init()
        self._lock = Lock()
        self.started = False
        if is_chief:
            self._httpd = BaseHTTPServer.HTTPServer((FLAGS.coord_host, FLAGS.coord_port), TrainingCoordinator.TrainingCoordinationHandler)

    def _reset_counters(self):
        self._index_train = 0
        self._index_dev = 0
        self._index_test = 0

    def _init(self):
        self._epochs_running = []
        self._epochs_done = []
        self._reset_counters()
        self._dev_losses = []

    def _log_all_jobs(self):
        '''Use this to debug-print epoch state'''
        log_debug('Epochs - running: %d, done: %d' % (len(self._epochs_running), len(self._epochs_done)))
        for epoch in self._epochs_running:
            log_debug('       - running: ' + epoch.job_status())

    def start_coordination(self, model_feeder, step=0):
        '''Starts to coordinate epochs and jobs among workers on base of
        data-set sizes, the (global) step and FLAGS parameters.

        Args:
            model_feeder (ModelFeeder): data-sets to be used for coordinated training

        Kwargs:
            step (int): global step of a loaded model to determine starting point
        '''
        with self._lock:
            self._init()

            # Number of GPUs per worker - fixed for now by local reality or cluster setup
            gpus_per_worker = len(available_devices)

            # Number of batches processed per job per worker
            batches_per_job  = gpus_per_worker * max(1, FLAGS.iters_per_worker)

            # Number of batches per global step
            batches_per_step = gpus_per_worker * max(1, FLAGS.replicas_to_agg)

            # Number of global steps per epoch - to be at least 1
            steps_per_epoch = max(1, model_feeder.train.total_batches // batches_per_step)

            # The start epoch of our training
            self._epoch = step // steps_per_epoch

            # Number of additional 'jobs' trained already 'on top of' our start epoch
            jobs_trained = (step % steps_per_epoch) * batches_per_step // batches_per_job

            # Total number of train/dev/test jobs covering their respective whole sets (one epoch)
            self._num_jobs_train = max(1, model_feeder.train.total_batches // batches_per_job)
            self._num_jobs_dev   = max(1, model_feeder.dev.total_batches   // batches_per_job)
            self._num_jobs_test  = max(1, model_feeder.test.total_batches  // batches_per_job)

            if FLAGS.epoch < 0:
                # A negative epoch means to add its absolute number to the epochs already computed
                self._target_epoch = self._epoch + abs(FLAGS.epoch)
            else:
                self._target_epoch = FLAGS.epoch

            # State variables
            # We only have to train, if we are told so and are not at the target epoch yet
            self._train = FLAGS.train and self._target_epoch > self._epoch
            self._test = FLAGS.test

            if self._train:
                # The total number of jobs for all additional epochs to be trained
                # Will be decremented for each job that is produced/put into state 'open'
                self._num_jobs_train_left = (self._target_epoch - self._epoch) * self._num_jobs_train - jobs_trained
                log_info('STARTING Optimization')
                self._training_time = stopwatch()

            # Important for debugging
            log_debug('step: %d' % step)
            log_debug('epoch: %d' % self._epoch)
            log_debug('target epoch: %d' % self._target_epoch)
            log_debug('steps per epoch: %d' % steps_per_epoch)
            log_debug('number of batches in train set: %d' % model_feeder.train.total_batches)
            log_debug('batches per job: %d' % batches_per_job)
            log_debug('batches per step: %d' % batches_per_step)
            log_debug('number of jobs in train set: %d' % self._num_jobs_train)
            log_debug('number of jobs already trained in first epoch: %d' % jobs_trained)

            self._next_epoch()

        # The coordinator is ready to serve
        self.started = True

    def _next_epoch(self):
        # State-machine of the coodination process

        # Indicates, if there were 'new' epoch(s) provided
        result = False

        # Make sure that early stop is enabled and validation part is enabled
        if (FLAGS.early_stop is True) and (FLAGS.validation_step > 0) and (len(self._dev_losses) >= FLAGS.earlystop_nsteps):

            # Calculate the mean of losses for past epochs
            mean_loss = np.mean(self._dev_losses[-FLAGS.earlystop_nsteps:-1])
            # Calculate the standard deviation for losses from validation part in the past epochs
            std_loss = np.std(self._dev_losses[-FLAGS.earlystop_nsteps:-1])
            # Update the list of losses incurred
            self._dev_losses = self._dev_losses[-FLAGS.earlystop_nsteps:]
            log_debug('Checking for early stopping (last %d steps) validation loss: %f, with standard deviation: %f and mean: %f' % (FLAGS.earlystop_nsteps, self._dev_losses[-1], std_loss, mean_loss))

            # Check if validation loss has started increasing or is not decreasing substantially, making sure slight fluctuations don't bother the early stopping from working
            if self._dev_losses[-1] > np.max(self._dev_losses[:-1]) or (abs(self._dev_losses[-1] - mean_loss) < FLAGS.estop_mean_thresh and std_loss < FLAGS.estop_std_thresh):
                # Time to early stop
                log_info('Early stop triggered as (for last %d steps) validation loss: %f with standard deviation: %f and mean: %f' % (FLAGS.earlystop_nsteps, self._dev_losses[-1], std_loss, mean_loss))
                self._dev_losses = []
                self._end_training()
                self._train = False

        if self._train:
            # We are in train mode
            if self._num_jobs_train_left > 0:
                # There are still jobs left
                num_jobs_train = min(self._num_jobs_train_left, self._num_jobs_train)
                self._num_jobs_train_left -= num_jobs_train

                # Let's try our best to keep the notion of curriculum learning
                self._reset_counters()

                # If the training part of the current epoch should generate a WER report
                is_display_step = FLAGS.display_step > 0 and (FLAGS.display_step == 1 or self._epoch > 0) and (self._epoch % FLAGS.display_step == 0 or self._epoch == self._target_epoch)
                # Append the training epoch
                self._epochs_running.append(Epoch(self._epoch, num_jobs_train, set_name='train', report=is_display_step))

                if FLAGS.validation_step > 0 and (FLAGS.validation_step == 1 or self._epoch > 0) and self._epoch % FLAGS.validation_step == 0:
                    # The current epoch should also have a validation part
                    self._epochs_running.append(Epoch(self._epoch, self._num_jobs_dev, set_name='dev', report=is_display_step))


                # Indicating that there were 'new' epoch(s) provided
                result = True
            else:
                # No jobs left, but still in train mode: concluding training
                self._end_training()
                self._train = False

        if self._test and not self._train:
            # We shall test, and are not in train mode anymore
            self._test = False
            self._epochs_running.append(Epoch(self._epoch, self._num_jobs_test, set_name='test', report=True))
            # Indicating that there were 'new' epoch(s) provided
            result = True

        if result:
            # Increment the epoch index - shared among train and test 'state'
            self._epoch += 1
        return result

    def _end_training(self):
        self._training_time = stopwatch(self._training_time)
        log_info('FINISHED Optimization - training time: %s' % format_duration(self._training_time))

    def start(self):
        '''Starts Training Coordinator. If chief, it starts a web server for
        communication with non-chief instances.
        '''
        if is_chief:
            log_debug('Starting coordinator...')
            self._thread = Thread(target=self._httpd.serve_forever)
            self._thread.daemon = True
            self._thread.start()
            log_debug('Coordinator started.')

    def stop(self, wait_for_running_epochs=True):
        '''Stops Training Coordinator. If chief, it waits for all epochs to be
        'done' and then shuts down the web server.
        '''
        if is_chief:
            if wait_for_running_epochs:
                while len(self._epochs_running) > 0:
                    log_traffic('Coordinator is waiting for epochs to finish...')
                    time.sleep(5)
            log_debug('Stopping coordinator...')
            self._httpd.shutdown()
            log_debug('Coordinator stopped.')

    def _talk_to_chief(self, path, data=None, default=None):
        tries = 0
        while tries < FLAGS.coord_retries:
            tries += 1
            try:
                url = 'http://%s:%d%s' % (FLAGS.coord_host, FLAGS.coord_port, path)
                log_traffic('Contacting coordinator - url: %s, tries: %d ...' % (url, tries-1))
                res = urllib.request.urlopen(urllib.request.Request(url, data, { 'content-type': 'text/plain' }))
                str = res.read()
                status = res.getcode()
                log_traffic('Coordinator responded - url: %s, status: %s' % (url, status))
                if status == 200:
                    return str
                if status == 204: # We use 204 (no content) to indicate end of training
                    return default
            except urllib.error.HTTPError as error:
                log_traffic('Problem reaching coordinator - url: %s, HTTP code: %d' % (url, error.code))
                pass
            time.sleep(10)
        return default

    def get_next_index(self, set_name):
        '''Retrives a new cluster-unique batch index for a given set-name.
        Prevents applying one batch multiple times per epoch.

        Args:
            set_name (str): name of the data set - one of 'train', 'dev', 'test'

        Returns:
            int. new data set index
        '''
        with self._lock:
            if is_chief:
                member = '_index_' + set_name
                value = getattr(self, member, -1)
                setattr(self, member, value + 1)
                return value
            else:
                # We are a remote worker and have to hand over to the chief worker by HTTP
                log_traffic('Asking for next index...')
                value = int(self._talk_to_chief(PREFIX_NEXT_INDEX + set_name))
                log_traffic('Got index %d.' % value)
                return value

    def _get_job(self, worker=0):
        job = None
        # Find first running epoch that provides a next job
        for epoch in self._epochs_running:
            job = epoch.get_job(worker)
            if job:
                return job
        # No next job found
        return None

    def get_job(self, worker=0):
        '''Retrieves the first job for a worker.

        Kwargs:
            worker (int): index of the worker to get the first job for

        Returns:
            WorkerJob. a job of one of the running epochs that will get
                associated with the given worker and put into state 'running'
        '''
        # Let's ensure that this does not interfer with other workers/requests
        with self._lock:
            if is_chief:
                # First try to get a next job
                job = self._get_job(worker)

                if job is None:
                    # If there was no next job, we give it a second chance by triggering the epoch state machine
                    if self._next_epoch():
                        # Epoch state machine got a new epoch
                        # Second try to get a next job
                        job = self._get_job(worker)
                        if job is None:
                            # Albeit the epoch state machine got a new epoch, the epoch had no new job for us
                            log_error('Unexpected case - no job for worker %d.' % (worker))
                        return job

                    # Epoch state machine has no new epoch
                    # This happens at the end of the whole training - nothing to worry about
                    log_traffic('No jobs left for worker %d.' % (worker))
                    self._log_all_jobs()
                    return None

                # We got a new job from one of the currently running epochs
                log_traffic('Got new %s' % job)
                return job

            # We are a remote worker and have to hand over to the chief worker by HTTP
            result = self._talk_to_chief(PREFIX_GET_JOB + str(FLAGS.task_index))
            if result:
                result = pickle.loads(result)
            return result

    def next_job(self, job):
        '''Sends a finished job back to the coordinator and retrieves in exchange the next one.

        Kwargs:
            job (WorkerJob): job that was finished by a worker and who's results are to be
                digested by the coordinator

        Returns:
            WorkerJob. next job of one of the running epochs that will get
                associated with the worker from the finished job and put into state 'running'
        '''
        if is_chief:
            # Try to find the epoch the job belongs to
            epoch = next((epoch for epoch in self._epochs_running if epoch.id == job.epoch_id), None)
            if epoch:
                # We are going to manipulate things - let's avoid undefined state
                with self._lock:
                    # Let the epoch finish the job
                    epoch.finish_job(job)
                    # Check, if epoch is done now
                    if epoch.done():
                        # If it declares itself done, move it from 'running' to 'done' collection
                        self._epochs_running.remove(epoch)
                        self._epochs_done.append(epoch)
                        log_info('%s' % epoch)
            else:
                # There was no running epoch found for this job - this should never happen.
                log_error('There is no running epoch of ID %d for job with ID %d. This should never happen.' % (job.epoch_id, job.id))
            return self.get_job(job.worker)

        # We are a remote worker and have to hand over to the chief worker by HTTP
        result = self._talk_to_chief('', data=pickle.dumps(job))
        if result:
            result = pickle.loads(result)
        return result

def initialize_globals():

    # ps and worker hosts required for p2p cluster setup
    FLAGS.ps_hosts = list(filter(len, FLAGS.ps_hosts.split(',')))
    FLAGS.worker_hosts = list(filter(len, FLAGS.worker_hosts.split(',')))

    # Determine, if we are the chief worker
    global is_chief
    is_chief = len(FLAGS.worker_hosts) == 0 or (FLAGS.task_index == 0 and FLAGS.job_name == 'worker')

    # Initializing and starting the training coordinator
    global COORD
    COORD = TrainingCoordinator()
    COORD.start()

    # The absolute number of computing nodes - regardless of cluster or single mode
    global num_workers
    num_workers = max(1, len(FLAGS.worker_hosts))

    # Create a cluster from the parameter server and worker hosts.
    global cluster
    cluster = tf.train.ClusterSpec({'ps': FLAGS.ps_hosts, 'worker': FLAGS.worker_hosts})

    # If replica numbers are negative, we multiply their absolute values with the number of workers
    if FLAGS.replicas < 0:
        FLAGS.replicas = num_workers * -FLAGS.replicas
    if FLAGS.replicas_to_agg < 0:
        FLAGS.replicas_to_agg = num_workers * -FLAGS.replicas_to_agg

    # The device path base for this node
    global worker_device
    worker_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task_index)

    # This node's CPU device
    global cpu_device
    cpu_device = worker_device + '/cpu:0'

    # This node's available GPU devices
    global available_devices
    available_devices = [worker_device + gpu for gpu in get_available_gpus()]

    # If there is no GPU available, we fall back to CPU based operation
    if 0 == len(available_devices):
        available_devices = [cpu_device]

    # Set default dropout rates
    if FLAGS.dropout_rate2 < 0:
        FLAGS.dropout_rate2 = FLAGS.dropout_rate
    if FLAGS.dropout_rate3 < 0:
        FLAGS.dropout_rate3 = FLAGS.dropout_rate
    if FLAGS.dropout_rate6 < 0:
        FLAGS.dropout_rate6 = FLAGS.dropout_rate

    global dropout_rates
    dropout_rates = [ FLAGS.dropout_rate,
                      FLAGS.dropout_rate2,
                      FLAGS.dropout_rate3,
                      FLAGS.dropout_rate4,
                      FLAGS.dropout_rate5,
                      FLAGS.dropout_rate6 ]

    global no_dropout
    no_dropout = [ 0.0 ] * 6

    # Set default checkpoint dir
    if len(FLAGS.checkpoint_dir) == 0:
        FLAGS.checkpoint_dir = xdg.save_data_path(os.path.join('deepspeech','checkpoints'))

    # Set default summary dir
    if len(FLAGS.summary_dir) == 0:
        FLAGS.summary_dir = xdg.save_data_path(os.path.join('deepspeech','summaries'))

    # Standard session configuration that'll be used for all new sessions.
    global session_config
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_placement)

    global alphabet
    alphabet = Alphabet(os.path.abspath(FLAGS.alphabet_config_path))

    # Geometric Constants
    # ===================

    # For an explanation of the meaning of the geometric constants, please refer to
    # doc/Geometry.md

    # Number of MFCC features
    global n_input
    n_input = 26 # TODO: Determine this programatically from the sample rate

    # The number of frames in the context
    global n_context
    n_context = 9 # TODO: Determine the optimal value using a validation data set

    # Number of units in hidden layers
    global n_hidden
    n_hidden = FLAGS.n_hidden

    global n_hidden_1
    n_hidden_1 = n_hidden

    global n_hidden_2
    n_hidden_2 = n_hidden

    global n_hidden_5
    n_hidden_5 = n_hidden

    # LSTM cell state dimension
    global n_cell_dim
    n_cell_dim = n_hidden

    # The number of units in the third layer, which feeds in to the LSTM
    global n_hidden_3
    n_hidden_3 = 2 * n_cell_dim

    # The number of characters in the target language plus one
    global n_character
    n_character = alphabet.size() + 1 # +1 for CTC blank label

    # The number of units in the sixth layer
    global n_hidden_6
    n_hidden_6 = n_character

    # Assign default values for standard deviation
    for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
        val = getattr(FLAGS, '%s_stddev' % var)
        if val is None:
            setattr(FLAGS, '%s_stddev' % var, FLAGS.default_stddev)

    # Queues that are used to gracefully stop parameter servers.
    # Each queue stands for one ps. A finishing worker sends a token to each queue befor joining/quitting.
    # Each ps will dequeue as many tokens as there are workers before joining/quitting.
    # This ensures parameter servers won't quit, if still required by at least one worker and
    # also won't wait forever (like with a standard `server.join()`).
    global done_queues
    done_queues = []
    for i, ps in enumerate(FLAGS.ps_hosts):
        # Queues are hosted by their respective owners
        with tf.device('/job:ps/task:%d' % i):
            done_queues.append(tf.FIFOQueue(1, tf.int32, shared_name=('queue%i' % i)))

    # Placeholder to pass in the worker's index as token
    global token_placeholder
    token_placeholder = tf.placeholder(tf.int32)

    # Enqueue operations for each parameter server
    global done_enqueues
    done_enqueues = [queue.enqueue(token_placeholder) for queue in done_queues]

    # Dequeue operations for each parameter server
    global done_dequeues
    done_dequeues = [queue.dequeue() for queue in done_queues]

    if len(FLAGS.one_shot_infer) > 0:
        FLAGS.train = False
        FLAGS.test = False
        FLAGS.export_dir = ''
        if not os.path.exists(FLAGS.one_shot_infer):
            log_error('Path specified in --one_shot_infer is not a valid file.')
            exit(1)

# Graph Creation
# ==============
def variable_on_worker_level(name, shape, initializer):
    r'''
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_worker_level()``
    used to create a variable in CPU memory.
    '''
    # Use the /cpu:0 device on worker_device for scoped operations
    if len(FLAGS.ps_hosts) == 0:
        device = worker_device
    else:
        device = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)

    with tf.device(device):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

def BiRNN(batch_x, seq_length, dropout):
    r'''
    That done, we will define the learned variables, the weights and biases,
    within the method ``BiRNN()`` which also constructs the neural network.
    The variables named ``hn``, where ``n`` is an integer, hold the learned weight variables.
    The variables named ``bn``, where ``n`` is an integer, hold the learned bias variables.
    In particular, the first variable ``h1`` holds the learned weight matrix that
    converts an input vector of dimension ``n_input + 2*n_input*n_context``
    to a vector of dimension ``n_hidden_1``.
    Similarly, the second variable ``h2`` holds the weight matrix converting
    an input vector of dimension ``n_hidden_1`` to one of dimension ``n_hidden_2``.
    The variables ``h3``, ``h5``, and ``h6`` are similar.
    Likewise, the biases, ``b1``, ``b2``..., hold the biases for the various layers.
    '''

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, n_input + 2*n_input*n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    b1 = variable_on_worker_level('b1', [n_hidden_1], tf.random_normal_initializer(stddev=FLAGS.b1_stddev))
    h1 = variable_on_worker_level('h1', [n_input + 2*n_input*n_context, n_hidden_1], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), FLAGS.relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

    # 2nd layer
    b2 = variable_on_worker_level('b2', [n_hidden_2], tf.random_normal_initializer(stddev=FLAGS.b2_stddev))
    h2 = variable_on_worker_level('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=FLAGS.h2_stddev))
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), FLAGS.relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

    # 3rd layer
    b3 = variable_on_worker_level('b3', [n_hidden_3], tf.random_normal_initializer(stddev=FLAGS.b3_stddev))
    h3 = variable_on_worker_level('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=FLAGS.h3_stddev))
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), FLAGS.relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell: (if else required for TF 1.0 and 1.1 compat)
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True) \
                   if 'reuse' not in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args else \
                   tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                input_keep_prob=1.0 - dropout[3],
                                                output_keep_prob=1.0 - dropout[3],
                                                seed=FLAGS.random_seed)
    # Backward direction cell: (if else required for TF 1.0 and 1.1 compat)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True) \
                   if 'reuse' not in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args else \
                   tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                input_keep_prob=1.0 - dropout[4],
                                                output_keep_prob=1.0 - dropout[4],
                                                seed=FLAGS.random_seed)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                             cell_bw=lstm_bw_cell,
                                                             inputs=layer_3,
                                                             dtype=tf.float32,
                                                             time_major=True,
                                                             sequence_length=seq_length)

    # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
    # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])

    # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    b5 = variable_on_worker_level('b5', [n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.b5_stddev))
    h5 = variable_on_worker_level('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.h5_stddev))
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), FLAGS.relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_worker_level('b6', [n_hidden_6], tf.random_normal_initializer(stddev=FLAGS.b6_stddev))
    h6 = variable_on_worker_level('h6', [n_hidden_5, n_hidden_6], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6], name="logits")

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6

def create_inference_graph(batch_size=None, output_is_logits=False, use_new_decoder=False):
    # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
    input_tensor = tf.placeholder(tf.float32, [batch_size, None, n_input + 2*n_input*n_context], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

    # Calculate the logits of the batch using BiRNN
    logits = BiRNN(input_tensor, tf.to_int64(seq_length) if FLAGS.use_seq_length else None, no_dropout)

    if output_is_logits:
        return logits

    # Beam search decode the batch
    #decoder = decode_with_lm if use_new_decoder else tf.nn.ctc_beam_search_decoder
    decoder = tf.nn.ctc_beam_search_decoder

    decoded, _ = decoder(logits, seq_length, merge_repeated=False, beam_width=FLAGS.beam_width)
    decoded = tf.convert_to_tensor(
        [tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in decoded], name='output_node')

    return (
        {
            'input': input_tensor,
            'input_lengths': seq_length,
        },
        {
            'outputs': decoded,
            'logits': logits,
        }
    )

def do_single_file_inference(input_file_path, checkpoint_dir):
    with tf.Session(config=session_config) as session:
        inputs, outputs = create_inference_graph(batch_size=1, use_new_decoder=True)

        # Create a saver using variables from the above newly created graph
        saver = tf.train.Saver(tf.global_variables())

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counterract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if not checkpoint:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.bind(('0.0.0.0', 6789))
        s.listen(1)
        while 1 :
            print("wait for client...wavefile")
            conn, _ = s.accept()
            with open('test/recording.wav', 'wb') as wavefile:
    			while True:
    				wavedata = conn.recv(1024)
    				if not wavedata :
    					break
    				wavefile.write(wavedata)
            #Recording(input_file_path)
            start_time = time.time()
            mfcc = audiofile_to_input_vector('test/'+input_file_path, n_input, n_context)

            output = session.run(outputs['outputs'], feed_dict = {
                inputs['input']: [mfcc],
                inputs['input_lengths']: [len(mfcc)],
            })
            text = ndarray_to_text(output[0][0], alphabet)
            conn, _ = s.accept()
            conn.send(text.encode('utf-8'))

def Recording(audio_file_path) :
    """recording module"""
    FORMAT=pyaudio.paInt16
    CHANNELS=1
    RATE=16000
    CHUNK=1024
    RECORD_SECONDS=10
    THRESHOLD=500
    FILE_NAME=audio_file_path

    print("Press a to start record and stop")

    while True :
        if keyboard.is_pressed('a') :
            break

    audio=pyaudio.PyAudio() #instantiate the pyaudio
    #recording prerequisites
    stream=audio.open(format=FORMAT,channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    #starting recording
    frames=[]
    for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
        data=stream.read(CHUNK)
        data_chunk=array('h',data)
        vol=max(data_chunk)
        if(vol>=THRESHOLD):
            frames.append(data)
        if keyboard.is_pressed('a') :
            break

    print("Stop recording")

    #end of recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    #writing to file
    wavfile=wave.open("test/"+audio_file_path, 'wb')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(audio.get_sample_size(FORMAT))
    wavfile.setframerate(RATE)
    wavfile.writeframes(b''.join(frames))#append frames recorded to file
    wavfile.close()
    print("Save wavefile")

if __name__ == '__main__' :
    initialize_globals()
    do_single_file_inference('recording.wav', '/media/kdy/2T/KDY/Speechdataset/checkpoint/190528/chk')
