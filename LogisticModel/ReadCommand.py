import sys
import getopt


#==============================================================================
# Command line processing
class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], "h",["guide=","output=", "guide_file="])
        opts = dict(opts)
        self.exit = True
        self.output = 'predictions.csv'
        if '-h' in opts:
            self.printHelp()
            return

        if '--guide_file' in opts:
            self.readfile = True
            self.filepath = opts['--guide_file']
            self.exit = False
            if '--output' in opts:
                self.output = opts['--output']

        elif '--guide' in opts:
            self.readfile = False
            self.guide = opts['--guide']
            self.exit = False
        else:
            self.printHelp()
            return
        
    def printHelp(self):
        print("xxx.py h [--guide_file filepath --output output_file_name --guide guide_sequence]\n")
        print("where:\n")
        print("\t h                  print help\n")
        print("\t[--guide_file filepath]    Optionally specifies the csv guide filepath 'filepath' to perform. \n")
        print("\t[--output outputpath]    Optionally specify the output file path when reading csv file.\n")
        print("\t[--guide guide_sequence]   Optionally specifies a guide sequence to get a prediction.\n")
        print("you must choose only one argument from the two arguments [--guide_file filepath] and [--guide guide_sequence].\n")


###
### 其他所需函数,def或class
# 例
def model():
#    xxxx
     return



if __name__ == '__main__':
    config = CommandLine() # read the arguments
    if config.exit == False:
        if config.readfile == True:
            filepath = config.filepath 
            data = pd.read_csv(filepath, index_col=0)
            ####
            # 输入读文件相关代码
            # 例：
            # pre_data = model(data)
            pre_data.to_csv(config.output, sep=',') # save the file
        elif config.readfile == False:
            guides = config.guide 
            ####
            # 输入单个sequence预测的相关代码
            # 例：
            # pre = model(guides)
            print('the predicted label for sequence %s is %d'% (guides, pre))

            
