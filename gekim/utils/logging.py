from datetime import datetime
import sys 

class Logger:
    #TODO: add history with tags of the method that made those, eg system.log.history['simulator']
    #TODO: format log to have method: message
    def __init__(self, quiet=False, logfilename=None):
        self.quiet = quiet
        self.logfilename = logfilename
        self.file_handle = None
        if self.logfilename:
            try:
                self.file_handle = open(self.logfilename, 'a')  # Open the file in append mode
            except IOError as e:
                print(f"Failed to open log file {self.logfilename}: {e}", file=sys.stderr)
                self.file_handle = None
        self._log(f"{datetime.now()} -- Logging started", "INFO")

    def _log(self, log_message, level):
        if not (self.quiet or level == "WARNING") and level != "ERROR":
            print(log_message)  # end='' to avoid double newlines
        if self.file_handle:
            try:
                self.file_handle.write(f"{log_message}\n")
                self.file_handle.flush()
            except IOError as e:
                print(f"Failed to write to log file: {e}", file=sys.stderr)
        if level == "ERROR":
            raise Exception(log_message)

    def info(self, message):
        self._log(message, "INFO")

    def warning(self, message):
        self._log(message, "WARNING")
    
    def error(self, message):
        self._log(message, "ERROR")

    def close(self):
        if self.file_handle:
            try:
                self.file_handle.close()
            except IOError as e:
                print(f"Failed to close log file: {e}", file=sys.stderr)
            finally:
                self.file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
