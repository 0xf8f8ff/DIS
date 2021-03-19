from mrjob.job import MRJob

considered_location = [
    'India', 'China', 'Iran', 'South Korea', 'South Africa', 'Kenya',
    'Bangladesh', 'Sweden', 'Norway', 'Germany', 'Italy', 'United Kingdom',
    'Brazil', 'United States', 'Canada', 'Australia'
]
class preprocessDataset(MRJob):

    def mapper(self, _, line):
        row = line.split(',')
        if len(row) > 45 and row[2] in considered_location:
            cleaned_row = [row[2], row[3], row[44], row[5], row[39]]
            yield (None, cleaned_row)


if __name__ == '__main__':
    preprocessDataset.run()