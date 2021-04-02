from mrjob.job import MRJob
import csv
from mrjob.protocol import RawValueProtocol

class processFlights(MRJob):
    OUTPUT_PROTOCOL = RawValueProtocol

    def mapper_init(self):
        self.lookup_table = {}
        for icao, country in csv.reader(open("lookupfile.csv", "r")):
            self.lookup_table[icao] = country

    def mapper(self, _, line):
        row = line.split(',')
        # skip first line
        if row[0] != "callsign":
            relevant = False
            # replace ICAO codes of airports with country names
            if len(row) > 9:
                if row[5] in self.lookup_table:
                    relevant = True
                    row[5] = self.lookup_table[row[5]].replace("\t", "")
                if row[6] in self.lookup_table:
                    relevant = True
                    row[6] = self.lookup_table[row[6]].replace("\t", "")
                if relevant:
                    date = row[9].split(" ")[0]
                    key = ','.join([date, row[5], row[6]])
                    yield (None, key)

if __name__ == '__main__':
    processFlights.run()
