import sys
import logging

def erro_message_detail(error, error_details: sys):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_message = "Error occoured in python script name [{0}] line numer [{1}] error message [{2}]".format(
        file_name,
        line_no,
        str(error)
    )
    
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        ''' 
            What does super().__init__(error_message) do?
            ✅ It sends the error_message to the Exception class so that Python's default exception handling can
        '''
        super().__init__(error_message)
        self.error_message = erro_message_detail(error_message, error_details=error_details)
        
    def __str__(self):
        return self.error_message


# ! For checking purpose
# if __name__ == "__main__":
#     try:
#         a = 1/ 0
#     except Exception as e:
#         logging.info("Divide by zero error")
#         raise CustomException(e, sys)