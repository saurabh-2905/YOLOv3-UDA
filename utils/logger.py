import tensorflow as tf
import os
import numpy as np


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        #Check for existing versions
        version_list = os.listdir(log_dir)
        if version_list == []:
            ind = 0
        else:
            version = [int(ver.split("_")[1]) for ver in version_list]
            ind = np.array(version).max() + 1

        #os.makedirs(f'version_{ind}', exist_ok=True)
        #Write file in latest version folder
        logs_new = os.path.join(log_dir, f'version_{ind}') 
        self.writer = tf.summary.create_file_writer(logs_new)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        # self.writer.add_summary(summary, step)
        with self.writer.as_default():
            for tag, value in tag_value_pairs:
                tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

