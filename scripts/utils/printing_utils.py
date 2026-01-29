"""
Printing and visualization utilities for the pipeline.
Provides dependency-free progress bars and formatted output.
"""

import sys
import time
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SimpleProgressBar:
    """A simple, dependency-free progress bar."""
    
    def __init__(self, total: int, desc: str = "Processing", bar_length: int = 30):
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
        
    def update(self, n: int = 1, context: str = ""):
        """Update progress by n steps."""
        self.current += n
        self._print_bar(context)
        
    def _print_bar(self, context: str = ""):
        percent = min(1.0, self.current / self.total)
        filled_len = int(self.bar_length * percent)
        bar = "=" * filled_len + "-" * (self.bar_length - filled_len)
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        
        # Format: Description: [====--] 65% (13/20) [Time: 12s, 1.2 it/s] Context
        msg = (
            f"\r{self.desc}: [{bar}] {percent:.0%} "
            f"({self.current}/{self.total}) "
            f"[Time: {elapsed:.0f}s, {rate:.1f} it/s] {context}"
        )
        # Pad with spaces to clear previous text
        sys.stdout.write(msg.ljust(80))
        sys.stdout.flush()
        
    def close(self):
        """Finalize the progress bar."""
        sys.stdout.write("\n")
        sys.stdout.flush()

def print_header(title: str, level: int = 1):
    """Print a styled header."""
    width = 80
    if level == 1:
        print("\n" + "=" * width)
        print(title.center(width))
        print("=" * width)
    else:
        print("\n" + "-" * width)
        print(title)
        print("-" * width)

def print_success(message: str):
    """Print a success message."""
    print(f"\n✓ {message}")

def print_warning(message: str):
    """Print a warning message."""
    print(f"\n! WARNING: {message}")

def print_dataframe(df: pd.DataFrame, title: str = None):
    """Print a DataFrame nicely formatted."""
    if title:
        print_header(title, level=2)
    
    # Configure pandas display options strictly for printing
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.width', 1000,
        'display.colheader_justify', 'left'
    ):
        print(df.to_string(index=False))
