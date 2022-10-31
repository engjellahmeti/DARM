# @project Deviance Analysis by Means of Redescription Mining - Master Thesis 
# @author Engjëll Ahmeti
# @date 12/20/2020

from enum import Enum


class Print(Enum):
    BLACK = '\033[90m'
    RED = '\033[91m'  # print errors
    GREEN = '\033[92m'
    YELLOW = '\033[93m'  # print real results
    BLUE = '\033[94m'   # print __str__
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'  # print regular logs

    END = '\033[0m'

    def __call__(self, s: str) -> str:
        """Returns colored string"""
        return "%s%s%s" % (self, s, Print.END)

    def __str__(self):
        return self.value

    def print(self, s:str) -> None:
        """Prints to stdout"""
        print(self(s))
