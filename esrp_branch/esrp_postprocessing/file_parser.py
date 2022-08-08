#!/usr/bin/env python3

from os import path
import re

class Results_File:
    def __init__(self,filename,iterations,runtime,rec_time,comp_residual,rep_residual):
        self.filename      = filename
        self.iterations    = iterations
        self.runtime       = runtime
        self.rec_time      = rec_time
        self.comp_residual = comp_residual
        self.rep_residual  = rep_residual


class File_Parser:
    def __init__(self):
        self.re_iter = re.compile("Iterations needed.*")
        self.re_runtime = re.compile("Solving the linear system.*")
        self.re_comp_residual = re.compile("Final residual norm is.*")
        self.re_rep_residual =  re.compile("Reported residual norm.*")

    def read_file(self,filename):
        fp = open(filename, "r");
        lines = fp.readlines()
        fp.close()
        iterations=None

        try:
            for line in lines:
                stripped_line = line.strip()
                if(self.re_iter.match(stripped_line) is not None):
                    words = stripped_line.split(sep=" ")
                    iterations = int(words[2])
                if(self.re_runtime.match(stripped_line) is not None):
                    words = stripped_line.split(sep=" ")
                    runtime = float(words[4])
                    rec_time = float(words[9])
                if(self.re_comp_residual.match(stripped_line) is not None):
                    words = stripped_line.split(sep=" ")
                    comp_residual = float(words[4])
                if(self.re_rep_residual.match(stripped_line) is not None):
                    words = stripped_line.split(sep=" ")
                    rep_residual = float(words[3])

        except e:
            print ("Exception while processing file\n{}".format(filename))
            raise e

        if iterations is None:
            raise( IOError("Problem with file {}".format(filename)) )


        return Results_File(filename, iterations, runtime, rec_time, comp_residual, rep_residual)
