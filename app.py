# MingAnn imports
import pickle
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from fuzzywuzzy import fuzz
import pandas as pd
from flask import Flask, render_template, request
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)
global_food_name = None


def load_results():
    with open('results.pickle', 'rb') as file:
        my_dict = pickle.load(file)
    return my_dict


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has a file part
        if 'marketing_csv' not in request.files:
            return render_template('recommended_sale.html', error='No file part')

        file = request.files['marketing_csv']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('recommended_sale.html', error='No selected file')

        if file:
            filename = 'salesData.csv'
            file.save(filename)
            analysis()
            # return a message indicating success
            return render_template('recommended_sale.html', message='File uploaded successfully')

    return render_template('recommended_sale.html')


@app.route('/recommend')
def recommend():
    return render_template('recommended_sale.html')


@app.route('/up_cross', methods=['GET', 'POST'])
def up_cross():
    df = pd.read_csv('salesData.csv')
    # assume your data frame is called 'df' and contains a column called 'food_name'
    food_list = df['name'].unique().tolist()
    return render_template('up_cross.html', food_list=food_list)


def load_data():
    df = pd.read_csv('salesData.csv')
    df["orderDateTime"] = pd.to_datetime(df["orderDateTime"])
    df = df.drop_duplicates(subset='orderId')
    return df


@app.route('/plot', methods=['GET', 'POST'])
def plot():
    # food_name = request.args.get('food_name')
    global global_food_name
    if request.method == 'POST':
        # Get the food name from the form
        food_name = request.form['food_name']
        food_name = request.form.get('food_name')
        global_food_name = food_name

    graph_type = request.args.get('graph_type')
    if graph_type is None:
        graph_type = 'daily'
    # if food_name is None:
    # return 'Please enter a food name'
    df = load_data()
    food_sales = df[df['name'] == global_food_name]
    if food_sales.empty:
        return 'No sales data for this food item'
    fig, ax = plt.subplots()
    if graph_type == 'daily':
        daily_sales = food_sales.groupby(df['orderDateTime'].dt.date)[
            'totalAmount'].sum().plot(ax=ax)
        ax.set_title('Daily Sales')
        ax.set_xlabel('Day')
        ax.set_ylabel('Salas')
        plt.savefig('static/day_plot.png')
    elif graph_type == 'weekly':
        weekly_sales = food_sales.groupby(df['orderDateTime'].dt.isocalendar().week)[
            'totalAmount'].sum().plot(ax=ax)
        ax.set_title('Weekly Sales')
        ax.set_xlabel('Week')
        ax.set_ylabel('Salas')
        plt.savefig('static/week_plot.png')
    elif graph_type == 'monthly':
        monthly_sales = food_sales.groupby(df['orderDateTime'].dt.month)[
            'totalAmount'].sum().plot(ax=ax)
        ax.set_title('Monthly Sales')
        ax.set_xlabel('Month')
        ax.set_ylabel('Salas')
        plt.savefig('static/month_plot.png')

    plt.close(fig)
    food_list = df['name'].unique().tolist()
    return render_template('up_cross.html', graph_type=graph_type, food_list=food_list, name=global_food_name)


@app.route('/campaign', methods=['GET', 'POST'])
def campaign():
    my_dict = load_results()
    df = pd.read_csv('salesData.csv')
    if request.method == 'POST':
        # Get selected key from dropdown
        selected_key = request.form.get('key')
        # Get value of selected key from dictionary
        selected_value = my_dict[selected_key]
        data = []
        data2 = []
        for i, tpl in enumerate(selected_value):
            increment = i + 1
            first = tpl[0]
            rest = tpl[1:]
            data.append((increment, first, rest))
            increment2 = 0
        increment2 = 0
        for item in rest:
            # get the price of the food from the sales_data DataFrame

            price = df.query("name == @item")['amount'].values[0]
            # compare prices and add to list
            # get the price of the food in list1, or set it to 0 if it is not in list1
            list1_price = df.query("name == @first")['amount'].values[0]

            # compare the prices of the two foods
            if price > list1_price:
                increment2 = increment2 + 1
                data2.append((increment2, first, rest))

        return render_template('market_product.html', my_dict=my_dict, selected_key=selected_key, data1=data, data2=data2)
    else:
        # Render template with dropdown populated with dictionary keys
        return render_template('market_product.html', my_dict=my_dict, selected_key=None, selected_value=None)


@app.route('/dict')
def dict():
    # Load the dictionary from the pickle file
    with open('my_dict.pickle', 'rb') as f:
        my_dict = pickle.load(f)

    df = pd.DataFrame(list(my_dict.items()), columns=[
                      'Items', 'Lists of Products Being Referenced'])
    table = df.to_html(index=False)
    return render_template('dict.html', table=table)

    # Render the template with the dictionary contents
    # return render_template('dict.html', dict_contents=dict_contents)


def analysis():
    df = pd.read_csv('salesData.csv')
    food_list = df['name'].unique().tolist()
    grs = list()  # groups of names with distance > 80
    for name in food_list:
        for g in grs:
            if all(fuzz.token_sort_ratio(name, w) > 80 for w in g):
                g.append(name)
                break
        else:
            grs.append([name, ])

    my_dict = {}
    for sublist in grs:
        key = sublist[0]
        value = sublist
        my_dict[key] = value

    # Open a file in binary mode and save the dictionary as a pickle file
    with open('my_dict.pickle', 'wb') as f:
        pickle.dump(my_dict, f)
    df1 = df
    # Loop through each value in the DataFrame
    for i, row in df.iterrows():
        value = row['name']
        for lst in grs:
            if value in lst:
                df.loc[i, 'name'] = lst[0]
                break  # Stop searching after finding the first matching list

    basket = (df.groupby(['orderId', 'name'])['quantity'].sum(
    ).unstack().reset_index().fillna(0).set_index('orderId'))
    encode_basket = basket.applymap(encode_units)
    filter_basket = encode_basket[(encode_basket > 0).sum(axis=1) >= 2]
    res = fpgrowth(filter_basket, min_support=0.01,
                   use_colnames=True).sort_values('support', ascending=False)
    res = association_rules(res, metric="lift", min_threshold=1).sort_values(
        'lift', ascending=False).reset_index(drop=True)
    res['food_pairs'] = [frozenset.union(
        *X) for X in res[['antecedents', 'consequents']].values]
    # Convert tuples to frozensets so we can sort them
    res['food_pairs'] = res['food_pairs'].apply(lambda x: frozenset(sorted(x)))

    # Drop duplicates, including those with reversed tuples
    res = res.drop_duplicates(subset=['food_pairs'])

    # Convert frozensets back to tuples
    res['food_pairs'] = res['food_pairs'].apply(lambda x: tuple(sorted(x)))

    # Convert the frozenset column to a tuple
    food_pairs = res['food_pairs'].apply(lambda x: tuple(x))
    food_pairs = food_pairs.drop_duplicates()

    # Group the foods by shop
    grouped = df.groupby('name.1')['name'].apply(list)

    # Initialize an empty dictionary to store the results
    results = {}

    # Iterate over the shops
    for shop, foods in grouped.items():
        # Check if any of the food pairs are in the shop
        pairs = [pair for pair in food_pairs if all(
            food in foods for food in pair)]
        # If there are any pairs, add them to the results dictionary
        if pairs:
            results[shop] = pairs

    # group food names by shop into a dictionary
    original_name = {}
    for shop, group in df1.groupby('name.1'):
        original_name[shop] = list(group['name'])

    for shop, items in results.items():
        for i, item in enumerate(items):
            new_item = list(item)
            for j, food in enumerate(item):
                if food in my_dict:
                    original_food_name = original_name[shop][j]
                    new_food_name = my_dict[food][0]
                    if new_food_name in original_name[shop]:
                        new_item[j] = new_food_name
                    else:
                        new_item[j] = original_food_name
            items[i] = tuple(new_item)
        results[shop] = items

    with open('results.pickle', 'wb') as f:
        pickle.dump(results, f)


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
