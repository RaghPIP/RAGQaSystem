import sys


class customexception(Exception):

    def __init__(self,error_message,error_details:sys): #sys module for exection detail - tracing complete execution
        self.error_message=error_message
        _,_,exc_tb=error_details.exc_info()
        print(exc_tb)

        self.lineno=exc_tb.tb_lineno #extract line number
        self.file_name=exc_tb.tb_frame.f_code.co_filename #extract file_name  

    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message))


if __name__=="__main__":
    try:
        a=1/0

    except Exception as e:
        #print(e)
        raise customexception(e,sys)