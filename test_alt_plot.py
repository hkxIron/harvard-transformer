import altair as alt
import pandas as pd

def test_alt1():
    #df = pd.read_csv('data/uk-election-results.csv')
    # import altair with an abbreviated alias
    import altair as alt

    # load a sample dataset as a pandas DataFrame
    from vega_datasets import data #pip install vega_datasets
    cars = data.cars()
    print(cars.columns)
    print(cars.head(3))

    # make the chart
    chart = alt.Chart(cars).mark_point().encode(
        x='Horsepower',
        y='Miles_per_Gallon',
        color='Origin',
    ).interactive()
    chart.save("data/house_power.html")

    alt.Chart(cars).mark_point().encode(
        x='Acceleration:Q',
        y='Miles_per_Gallon:Q',
        color='Origin:N'
    ).save("data/horse_power2.html")

    alt.Chart(cars).mark_point().encode(
        alt.X('Acceleration', type='quantitative'),
        alt.Y('Miles_per_Gallon', type='quantitative'),
        alt.Color('Origin', type='nominal')
    ).save("data/horse_power3.html")

if __name__ == '__main__':
    test_alt1()
