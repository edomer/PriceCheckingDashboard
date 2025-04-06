import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
import zipfile
import threading
import requests
from requests.exceptions import RequestException
from contextlib import contextmanager


# Set page config
st.set_page_config(
    page_title="Árstatisztika Vizualizáció",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================
# Thread-safe DataModel class (with all methods)
# =============================================

class DataModel:
    def __init__(self, db_name='prices.db'):
        self.db_name = db_name
        self.lock = threading.Lock()
        self._initialize_database()

    @contextmanager
    def _get_connection(self):
        """Thread-safe connection context manager"""
        conn = sqlite3.connect(self.db_name)
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    product_id INTEGER,
                    year INTEGER,
                    month INTEGER,
                    price REAL,
                    PRIMARY KEY (product_id, year, month),
                    FOREIGN KEY (product_id) REFERENCES products(id)
                )
            """)
            conn.commit()

    def import_csv(self, csv_path):
        """Import data from CSV file with Hungarian month handling"""
        month_map = {
            'január': 1, 'február': 2, 'március': 3, 'április': 4,
            'május': 5, 'június': 6, 'július': 7, 'augusztus': 8,
            'szeptember': 9, 'október': 10, 'november': 11, 'december': 12
        }

        try:
            # Try different encodings for Hungarian text
            try:
                df = pd.read_csv(csv_path, header=1, encoding='ISO-8859-2', delimiter=';')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, header=1, encoding='Windows-1250', delimiter=';')

            df.replace("..", None, inplace=True)

            # Detect if monthly data (columns contain dots like "2021.január")
            is_monthly = any('.' in col for col in df.columns[2:])

            with self._get_connection() as conn:
                cursor = conn.cursor()

                for _, row in df.iterrows():
                    product_id = int(row.iloc[0])
                    product_name = str(row.iloc[1]).strip()

                    # Insert product
                    cursor.execute(
                        "INSERT OR IGNORE INTO products (id, name) VALUES (?, ?)",
                        (product_id, product_name)
                    )

                    # Process each price column
                    for col in df.columns[2:]:
                        price = row[col]
                        if pd.isna(price) or price == "..":
                            continue

                        try:
                            price = float(str(price).replace(" ", ""))
                        except:
                            continue

                        if is_monthly:
                            try:
                                year_part, month_part = col.split('.')
                                year = int(year_part)
                                month_name = month_part.strip().lower()
                                month = month_map.get(month_name)
                                if not month:
                                    continue
                            except:
                                continue
                        else:
                            try:
                                year = int(col.strip())
                                month = None
                            except:
                                continue

                        cursor.execute(
                            "INSERT OR REPLACE INTO prices (product_id, year, month, price) VALUES (?, ?, ?, ?)",
                            (product_id, year, month, price)
                        )

                conn.commit()
            return True
        except Exception as e:
            st.error(f"Hiba történt az importálás során: {str(e)}")
            return False

    def switch_database(self, db_name):
        """Switch to a different database file"""
        self.db_name = db_name

    def get_prices(self, product_ids, start_date, end_date, is_monthly):
        """Get prices for given products and date range - fixed version"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            prices = []

            try:
                if is_monthly:
                    start_year, start_month = map(int, start_date.split('-'))
                    end_year, end_month = map(int, end_date.split('-'))

                    # Simplified monthly query
                    query = """
                        SELECT year, month, price 
                        FROM prices
                        WHERE product_id = ?
                        AND (
                            (year = ? AND month >= ?)
                            OR (year > ? AND year < ?)
                            OR (year = ? AND month <= ?)
                        )
                        ORDER BY year, month
                    """
                    params = (product_ids[0], start_year, start_month,
                              start_year, end_year, end_year, end_month)

                else:
                    start_year = int(start_date)
                    end_year = int(end_date)

                    # Yearly query (works as before)
                    query = """
                        SELECT year, month, price 
                        FROM prices
                        WHERE product_id = ? 
                        AND year BETWEEN ? AND ?
                        ORDER BY year, month
                    """
                    params = (product_ids[0], start_year, end_year)

                cursor.execute(query, params)
                prices = [(product_ids[0], row[0], row[1], row[2]) for row in cursor.fetchall()]

            except Exception as e:
                st.error(f"Database query error: {str(e)}")
                return []

            return prices

    def get_products_table_data(self, start_date=None, end_date=None, sort_method=None, ascending=True):
        """Get products with additional metrics for table display"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get basic product info
            cursor.execute("SELECT id, name FROM products")
            products = [{'id': row[0], 'name': row[1]} for row in cursor.fetchall()]

            # If no date range provided, just return basic info
            if not start_date or not end_date:
                return pd.DataFrame(products)

            # Initialize is_monthly before any validation
            is_monthly = '-' in start_date

            # Validate that both dates have the same format
            if ('-' in start_date) != ('-' in end_date):
                st.warning("A dátumformátumoknak meg kell egyezniük (ÉÉÉÉ vagy ÉÉÉÉ-HH)!")
                return pd.DataFrame(products)

            # Date validation
            try:
                if is_monthly:
                    start_year, start_month = map(int, start_date.split('-'))
                    end_year, end_month = map(int, end_date.split('-'))

                    # Validate month values
                    if not (1 <= start_month <= 12) or not (1 <= end_month <= 12):
                        st.warning("Érvénytelen hónap érték (1-12 között kell legyen)!")
                        return pd.DataFrame(products)

                    # Convert to comparable strings
                    start_str = f"{start_year:04d}-{start_month:02d}"
                    end_str = f"{end_year:04d}-{end_month:02d}"

                    if start_str > end_str:
                        st.warning("A kezdő dátum nem lehet későbbi, mint a záró dátum!")
                        return pd.DataFrame(products)

                else:
                    start_year = int(start_date)
                    end_year = int(end_date)

                    if start_year > end_year:
                        st.warning("A kezdő év nem lehet későbbi, mint a záró év!")
                        return pd.DataFrame(products)

            except ValueError:
                st.warning("Érvénytelen dátum formátum! Használjon ÉÉÉÉ vagy ÉÉÉÉ-HH formátumot.")
                return pd.DataFrame(products)

            # Calculate metrics for each product
            for product in products:
                if is_monthly:
                    query = """
                        SELECT year, month, price FROM prices 
                        WHERE product_id = ? AND 
                        ((year > ?) OR (year = ? AND month >= ?)) AND 
                        ((year < ?) OR (year = ? AND month <= ?))
                        ORDER BY year, month
                    """
                    params = (product['id'], start_year, start_year, start_month,
                              end_year, end_year, end_month)
                else:
                    query = """
                        SELECT year, month, price FROM prices 
                        WHERE product_id = ? AND year BETWEEN ? AND ?
                        ORDER BY year, month
                    """
                    params = (product['id'], start_year, end_year)

                cursor.execute(query, params)
                prices = cursor.fetchall()

                if not prices:
                    product.update({
                        'start_price': None,
                        'end_price': None,
                        'price_diff': None,
                        'avg_price': None
                    })
                    continue

                # Calculate metrics
                start_price = next((p[2] for p in prices if
                                    (p[0] == start_year and
                                     (not is_monthly or p[1] == start_month))), None)
                end_price = next((p[2] for p in prices if
                                  (p[0] == end_year and
                                   (not is_monthly or p[1] == end_month))), None)

                price_diff = end_price - start_price if start_price and end_price else None
                valid_prices = [p[2] for p in prices if p[2] is not None]
                avg_price = sum(valid_prices) / len(valid_prices) if valid_prices else None

                product.update({
                    'start_price': start_price,
                    'end_price': end_price,
                    'price_diff': price_diff,
                    'avg_price': avg_price
                })

                # Convert to DataFrame
                df = pd.DataFrame(products)

                # Apply sorting if specified
                if sort_method:
                    if sort_method == "Legnagyobb árkülönbség":
                        df = df.sort_values('price_diff', ascending=not ascending, na_position='last')
                    elif sort_method == "Legkisebb árkülönbség":
                        df = df.sort_values('price_diff', ascending=ascending, na_position='last')
                    elif sort_method == "Átlagár a kiválasztott években": \
                            df = df.sort_values('avg_price', ascending=not ascending, na_position='last')

            return df

    def get_available_dates(self):
        """Get all available years or year-month combinations"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Check if we have monthly data
            cursor.execute("SELECT COUNT(DISTINCT month) FROM prices WHERE month IS NOT NULL")
            has_monthly = cursor.fetchone()[0] > 0

            if has_monthly:
                cursor.execute("""
                    SELECT DISTINCT year, month 
                    FROM prices 
                    WHERE price IS NOT NULL
                    ORDER BY year, month
                """)
                return [f"{row[0]}-{row[1]:02d}" for row in cursor.fetchall()]
            else:
                cursor.execute("""
                    SELECT DISTINCT year 
                    FROM prices 
                    WHERE price IS NOT NULL
                    ORDER BY year
                """)
                return [str(row[0]) for row in cursor.fetchall()]


# =============================================
# Streamlit UI Implementation
# =============================================

# Initialize the model in session state
if 'model' not in st.session_state:
    st.session_state.model = DataModel()

# Initialize table data in session state
if 'products_table' not in st.session_state:
    try:
        st.session_state.products_table = st.session_state.model.get_products_table_data()
    except Exception as e:
        st.session_state.products_table = pd.DataFrame(columns=['id', 'name'])
        st.warning("No products found in database. Please import data first.")

# SIDEBAR - Data management and database selection
with st.sidebar:
    st.header("Adatkezelés")

    with st.expander("Adatbázis kezelése", expanded=False):
        # Database selection dropdown
        db_files = [f for f in os.listdir() if f.endswith('.db')]
        selected_db = st.selectbox("Aktív adatbázis", db_files, key='db_selector')
        if selected_db and selected_db != st.session_state.model.db_name:
            st.session_state.model.switch_database(selected_db)
            st.session_state.products_table = st.session_state.model.get_products_table_data()

        # File uploader for manual CSV import
        uploaded_file = st.file_uploader("Kézi CSV importálás", type=['csv'])
        if uploaded_file is not None:
            with st.spinner("Importálás folyamatban..."):
                with open("temp_upload.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Use the exact same name as CSV but with .db extension
                new_db_name = os.path.splitext(uploaded_file.name)[0] + ".db"
                new_model = DataModel(new_db_name)

                if new_model.import_csv("temp_upload.csv"):
                    st.success("Sikeres importálás!")
                    st.session_state.model = new_model
                    st.session_state.products_table = new_model.get_products_table_data()
                    os.remove("temp_upload.csv")
                    st.rerun()
                else:
                    st.error("Importálási hiba!")

        # Single update button with fixed functionality
        if st.button("Adatok frissítése a KSH-ról", key='ksh_update'):
            with st.spinner("KSH adatok letöltése..."):
                try:
                    ksh_url = "https://www.ksh.hu/stadat_files/ara/hu/ara0044.csv"
                    db_name = "stadat-ara0044-1.2.1.8-hu_autoupdated_web.db"

                    # Download the file with timeout
                    response = requests.get(ksh_url, timeout=10)
                    response.raise_for_status()

                    # Save temporary file
                    temp_csv = "ksh_latest.csv"
                    with open(temp_csv, "wb") as f:
                        f.write(response.content)

                    # Create and import to new database
                    new_model = DataModel(db_name)
                    if new_model.import_csv(temp_csv):
                        st.success(f"Sikeres frissítés! Adatbázis: {db_name}")
                        st.session_state.model = new_model
                        st.session_state.products_table = new_model.get_products_table_data()
                        os.remove(temp_csv)
                        st.rerun()
                    else:
                        st.error("Hiba az adatbázis létrehozásakor!")

                except requests.exceptions.RequestException as e:
                    st.error(f"Hiba a letöltéskor: {str(e)}")
                except Exception as e:
                    st.error(f"Váratlan hiba: {str(e)}")

# Create main columns (100% width for the right column now)
left_col, right_col = st.columns([4, 6], gap="medium")

# LEFT COLUMN - Product table with search and sorting
with left_col:
    st.header("Terméklista")

    # Sorting options
    st.subheader("Rendezés")

    # Date range for sorting
    dates = st.session_state.model.get_available_dates()
    if dates:
        sort_start, sort_end = st.select_slider(
            "Rendezési időszak",
            options=dates,
            value=(dates[0], dates[-1]))

        # Sorting method selection - Modified to have first item as default
        sort_method = st.selectbox(
            "Rendezési szempont",
            ["Legnagyobb árkülönbség", "Legkisebb árkülönbség", "Átlagár a kiválasztott években"],
            index=0
        )

        # Sort order toggle
        if sort_method:  # Only show if a method is selected
            # Initialize in session state if not exists
            if 'sort_ascending' not in st.session_state:
                st.session_state.sort_ascending = True

            # Create the toggle with dynamic label
            sort_ascending = st.toggle(
                "Csökkenő sorrend" if st.session_state.sort_ascending else "Növekvő sorrend",
                value=st.session_state.sort_ascending,
                key="sort_order_toggle"
            )

            # Update session state if changed
            if sort_ascending != st.session_state.sort_ascending:
                st.session_state.sort_ascending = sort_ascending
                st.rerun()  # This will refresh to show the new label


        # Sort button
        if st.button("Rendez") and sort_method:
            st.session_state.products_table = st.session_state.model.get_products_table_data(
                sort_start, sort_end, sort_method, sort_ascending)

        # Change the search text input to have placeholder text
        search_term = st.text_input("Keresés terméknév alapján", "", placeholder="Keresés")

        # Display filtered and sorted table
        if search_term:
            display_df = st.session_state.products_table[
                st.session_state.products_table['name'].str.contains(search_term, case=False)]
        else:
            display_df = st.session_state.products_table

        # Format the table display
        formatted_df = display_df.copy()
        formatted_df = display_df.copy()
        if 'start_price' in formatted_df.columns:
            formatted_df['start_price'] = formatted_df['start_price'].apply(
                lambda x: f"{int(x):,} Ft".replace(",", " ") if pd.notnull(x) else "-")
        if 'end_price' in formatted_df.columns:
            formatted_df['end_price'] = formatted_df['end_price'].apply(
                lambda x: f"{int(x):,} Ft".replace(",", " ") if pd.notnull(x) else "-")
        if 'price_diff' in formatted_df.columns:
            formatted_df['price_diff'] = formatted_df['price_diff'].apply(
                lambda x: f"{int(x):,} Ft".replace(",", " ") if pd.notnull(x) else "-")
        if 'avg_price' in formatted_df.columns:
            formatted_df['avg_price'] = formatted_df['avg_price'].apply(
                lambda x: f"{int(x):,} Ft".replace(",", " ") if pd.notnull(x) else "-")

        # Display the table
        st.dataframe(
            formatted_df,
            column_config={
                "id": None,
                "name": "Terméknév",
                "start_price": "Kezdő ár",
                "end_price": "Záró ár",
                "price_diff": "Árkülönbség",
                "avg_price": "Átlagár"},
            hide_index=True,
            use_container_width=True,
            height=600)

# RIGHT COLUMN - Visualization only
with right_col:
    # Visualization section
    st.header("Ábrázolás")

    # Get selected products from table
    if not st.session_state.products_table.empty:
        selected_indices = st.multiselect(
            "Válasszon termékeket az ábrázoláshoz",
            options=st.session_state.products_table['name'].tolist(),
            default=st.session_state.products_table['name'].iloc[0] if len(
                st.session_state.products_table) > 0 else None
        )

    # Date range selection for visualization
    if dates:
        viz_start, viz_end = st.select_slider(
            "Ábrázolási időszak",
            options=dates,
            value=(dates[0], dates[-1]))

        # Plot type selection
        plot_type = st.selectbox(
            "Ábrázolás típusa",
            ["Vonaldiagram", "Pontdiagram", "Oszlopdiagram", "Területdiagram"],
            index=0
        )

        # Analysis button
        if st.button("Elemzés indítása") and selected_indices and dates:
            try:
                product_ids = st.session_state.products_table[
                    st.session_state.products_table['name'].isin(selected_indices)
                ]['id'].tolist()

                is_monthly = '-' in viz_start

                # Get prices for each product individually
                all_prices = []
                for pid in product_ids:
                    prices = st.session_state.model.get_prices(
                        [pid],
                        viz_start,
                        viz_end,
                        is_monthly
                    )
                    all_prices.extend(prices)

                if not all_prices:
                    st.warning("Nincsenek adatok a kiválasztott tartományban a kiválasztott termék(ek)hez.")
                else:
                    # Convert to DataFrame for plotting
                    df = pd.DataFrame(all_prices, columns=['product_id', 'year', 'month', 'price'])
                    df['product'] = df['product_id'].map(
                        st.session_state.products_table.set_index('id')['name'].to_dict()
                    )

                    # Create date string for display
                    if is_monthly:
                        df['date_str'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
                    else:
                        df['date_str'] = df['year'].astype(str)

                    # Create plot based on selected type
                    fig = None
                    if plot_type == "Vonaldiagram":
                        fig = px.line(
                            df,
                            x='date_str',
                            y='price',
                            color='product',
                            title="Árváltozás időbeli alakulása",
                            labels={'price': 'Ár (Ft)', 'date_str': 'Dátum'}
                        )
                    elif plot_type == "Pontdiagram":
                        fig = px.scatter(
                            df,
                            x='date_str',
                            y='price',
                            color='product',
                            title="Árváltozás időbeli alakulása",
                            labels={'price': 'Ár (Ft)', 'date_str': 'Dátum'}
                        )
                    elif plot_type == "Oszlopdiagram":
                        fig = px.bar(
                            df,
                            x='date_str',
                            y='price',
                            color='product',
                            barmode='group',
                            title="Árváltozás időbeli alakulása",
                            labels={'price': 'Ár (Ft)', 'date_str': 'Dátum'}
                        )
                    elif plot_type == "Területdiagram":
                        fig = px.area(
                            df,
                            x='date_str',
                            y='price',
                            color='product',
                            title="Árváltozás időbeli alakulása",
                            labels={'price': 'Ár (Ft)', 'date_str': 'Dátum'}
                        )

                    if fig:
                        # Create formatted date string with dots (only used for x-axis)
                        if is_monthly:
                            df['formatted_date'] = df['year'].astype(str) + '.' + df['month'].astype(str).str.zfill(
                                2) + '.'
                        else:
                            df['formatted_date'] = df['year'].astype(str) + '.'  # Just year with dot

                        # Custom hover template WITHOUT date (since it's already visible on x-axis)
                        hovertemplate = (
                                "<b>%{fullData.name}</b><br>" +  # Product name only
                                "Ár: %{y:,.0f} Ft".replace(",", " ") +  # Space as a thousand separator
                                "<extra></extra>"  # Formatted price
                        )

                        # Apply to all traces
                        for trace in fig.data:
                            trace.update(
                                hovertemplate=hovertemplate,
                                hoverinfo='skip',
                                hoverlabel=dict(
                                    bgcolor="white",
                                    font_size=14,
                                    font_family="Arial"
                                )
                            )

                        # Update x-axis to show formatted dates
                        if is_monthly:
                            fig.update_xaxes(
                                tickformat="%Y.%m."  # Shows as "2023.03." on axis
                            )
                        else:
                            fig.update_xaxes(
                                tickformat="%Y."  # Shows as "2023." on axis
                            )

                        # Rest of layout config
                        fig.update_layout(
                            hovermode='x unified',
                            xaxis_title='Dátum',
                            yaxis_title='Ár (Ft)',
                            legend_title='Termékek',
                            height=600,
                            separators="., "  # This sets space as thousand separator
                        )

                        st.plotly_chart(fig, use_container_width=True)


            except Exception as e:
                st.error(f"Hiba történt az ábrázolás során: {str(e)}")