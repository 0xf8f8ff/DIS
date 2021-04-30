from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol
import csv

class flight_columns(MRJob):
    OUTPUT_PROTOCOL = RawValueProtocol

    # read list of the selected countries
    def mapper_init(self):
        self.locations = list()
        for country in csv.reader(open("country_list", "r")):
            self.locations.append(country)

    # yield format [internal, international]
    def mapper(self, _, line):
        row = line.split(',')
        # ignore too short lines or lines with empty destination, date
        if len(row) > 2 and row[0] != "" and row[2].replace("\t", "") in self.locations:
            key = ','.join([row[0], row[2].replace("\t", "")])
            if row[1] == row[2].replace("\t", ""):
                yield (key, [1,0])
            else:
                yield (key, [0,1])

    def reducer(self, key, values):
        inflights = 0
        intflights = 0
        for val in values:
            inflights += val[0]
            intflights += val[1]
        line = ','.join([key, str(inflights), str(intflights)])
        yield (None, line)

if __name__ == '__main__':
    flight_columns.run()

