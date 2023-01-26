class NetMinerNumError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self)  # 初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo
class NetAdjError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self)  # 初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo