from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol
import csv

airport_types = [
   "\"large_airport\"", "\"small_airport\"", "\"medium_airport\""
]

class joinISOcodes(MRJob):
    OUTPUT_PROTOCOL = RawValueProtocol

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init_ISO,
                mapper=self.mapper_ISO,
                reducer=self.reducer_ISO),
            MRStep(mapper=self.mapper_ICAO)
            ]

    def mapper_init_ISO(self):
        self.locations = list()
        for country in csv.reader(open("country_list", "r")):
            self.locations.append(country)

    def mapper_ISO(self, _, line):
        row = line.split(',')
        # skip first line
        if row[0] != "id":
            # the row is from the ICAO code dataset
            if len(row) >= 9:
                if row[2] in airport_types:
                    yield (row[8].replace('"', ''), row[1].replace('"', ''))
            # the row is from country ISO dataset
            elif len(row) > 4 and len(row[0]) == 6:
                yield (row[1].replace('"', ''), row[2].replace('"', ''))

    def reducer_ISO(self, key, values):
        yield key, list(values)

    def mapper_ICAO(self, key, codes):
        # ignore empty keys
        if key != "" and len(key) > 2:
            if len(codes) > 1: 
                countryName = ""
                for code in codes:
                    if code in self.locations:
                        countryName = code
                if countryName != "":
                    for code in codes:
                        if code != countryName:           
                            line = ','.join([code, countryName])
                            yield (None, line)

if __name__ == '__main__':
    joinISOcodes.run()
