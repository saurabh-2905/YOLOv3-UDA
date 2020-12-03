#import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
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
        # Write log files for train and val
        #self.writer = tf.summary.create_file_writer(logs_new)
        self.writer = SummaryWriter(log_dir = os.path.join(logs_new,'train'))
        self.writer_val = SummaryWriter(log_dir = os.path.join(logs_new, 'val'))

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""    
        # with self.writer.as_default():
        #     tf.summary.scalar(tag, value, step=step)
        #     self.writer.flush()
        
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        # with self.writer.as_default():
        #     for tag, value in tag_value_pairs:
        #         tf.summary.scalar(tag, value, step=step)
        #     self.writer.flush()
        
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

    def val_scalar_summary(self, tag, value, step):
        """Log a scalar variable."""    
        # with self.writer.as_default():
        #     tf.summary.scalar(tag, value, step=step)
        #     self.writer.flush()
        
        self.writer_val.add_scalar(tag, value, step)

    def val_list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        # with self.writer.as_default():
        #     for tag, value in tag_value_pairs:
        #         tf.summary.scalar(tag, value, step=step)
        #     self.writer.flush()
        
        for tag, value in tag_value_pairs:
            self.writer_val.add_scalar(tag, value, step)

    def val_plot_detection(self, tag, figure):
        """"Log Images of Detection"""
        self.writer_val.add_figure(tag, figure)
            

