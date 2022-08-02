import os,sys

class CreditException(Exception):
    def __init__(self,error_message:Exception,error_detail:sys):
        super().__init__(error_message)
        self.error_message=CreditException.get_error_message(error_message=error_message,
                                                            error_detail=error_detail)

    @staticmethod
    def get_error_message(error_message:Exception,error_detail:sys)->str:
        """error_message : Exception class object
        error_detail: object of sys module
        --------------------------------------------------------------
        Returns error_message (str)
        """
        _,_,exec_tb=error_detail.exc_info()
        exception_block_line_no=exec_tb.tb_frame.f_lineno
        try_block_lineno=exec_tb.tb_lineno
        file_name= exec_tb.tb_frame.f_code.co_filename
        error_message=f"""Error occured in script:[{file_name}] at
                            try block line number:[{try_block_lineno}]
                            and Exception block line number :[{exception_block_line_no}]
                            error message:[{error_message}]"""

        return error_message

    def __str__(self):
        return self.error_message

    def __repr__(self) -> str:
        return CreditException.__name__.str()

    