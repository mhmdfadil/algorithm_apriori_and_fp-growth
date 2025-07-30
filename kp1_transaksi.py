# Mengimpor library yang diperlukan
import pandas as pd  # Untuk manipulasi data dan dataframe
import json  # Untuk bekerja dengan format JSON
import os  # Untuk operasi sistem file
from itertools import chain  # Untuk menggabungkan iterables

def process_excel(input_file, output_folder):
    """
    Fungsi untuk memproses file Excel transaksi menjadi beberapa format
    
    Parameters:
        input_file (str): Path file Excel input
        output_folder (str): Folder tujuan untuk menyimpan output
        
    Returns:
        dict: Dictionary berisi path file-file output yang dihasilkan
    """
    
    # Membuat folder output jika belum ada
    # exist_ok=True agar tidak error jika folder sudah ada
    os.makedirs(output_folder, exist_ok=True)
    
    # Membaca file Excel input menggunakan pandas
    df = pd.read_excel(input_file)
    
    # ==============================================
    # TAHAP 1: MEMISAHKAN ITEM DALAM TRANSAKSI
    # ==============================================
    
    # Memisahkan string Item berdasarkan koma (dengan optional spasi setelahnya)
    # expand=True mengembalikan DataFrame dengan kolom terpisah
    items_split = df['Item'].str.split(r',\s*', expand=True)
    
    # Memberi nama kolom dengan format Item 1, Item 2, dst
    items_split.columns = [f'Item {i+1}' for i in range(items_split.shape[1])]
    
    # Menggabungkan kolom ID Transaksi dan Tahun dengan hasil pemisahan item
    result_df = pd.concat([df[['ID Transaksi', 'Tahun']], items_split], axis=1)
    
    # Menentukan path untuk file output
    processed_excel_path = os.path.join(output_folder, 'tahap1_transaksi_processed.xlsx')
    json_path = os.path.join(output_folder, 'tahap2_transaksi_basket.json')
    one_hot_path = os.path.join(output_folder, 'tahap3_one_hot_transaksi.xlsx')
    
    # Menyimpan hasil tahap 1 ke Excel
    with pd.ExcelWriter(processed_excel_path, engine='xlsxwriter') as writer:
        # Menulis dataframe ke sheet 'Transaksi'
        result_df.to_excel(writer, index=False, sheet_name='Transaksi')
        
        # Mengatur lebar kolom otomatis sesuai isinya
        worksheet = writer.sheets['Transaksi']
        for i, col in enumerate(result_df.columns):
            # Menghitung panjang maksimum isi kolom + buffer 2 karakter
            max_len = max(result_df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)
    
    # ==============================================
    # TAHAP 2: MEMBUAT FORMAT JSON BASKET
    # ==============================================
    
    # Membuat list basket dari kolom Item
    # Setiap transaksi menjadi list item yang sudah di-trim (dihilangkan spasi di awal/akhir)
    basket_list = [
        [item.strip() for item in str(row['Item']).split(',')] 
        for _, row in df.iterrows()
    ]
    
    # Menyimpan basket list ke file JSON dengan indentasi 4 spasi
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(basket_list, f, indent=4)
    
    # ==============================================
    # TAHAP 3: ONE-HOT ENCODING
    # ==============================================
    
    # Mendapatkan semua item unik dari seluruh transaksi
    # chain digunakan untuk menggabungkan semua list item menjadi satu list besar
    # set digunakan untuk mendapatkan item unik
    # sorted untuk mengurutkan secara alfabetis
    all_items = sorted(list(set(chain(*[str(items).split(', ') for items in df['Item']]))))
    
    # Membuat data one-hot encoding
    one_hot_data = []
    for items in df['Item']:
        # Membersihkan dan memisahkan item
        item_list = [item.strip() for item in str(items).split(',')]
        # Membuat baris one-hot (1 jika item ada di transaksi, 0 jika tidak)
        one_hot_row = [1 if item in item_list else 0 for item in all_items]
        one_hot_data.append(one_hot_row)
    
    # Membuat dataframe one-hot encoding
    one_hot_df = pd.DataFrame(one_hot_data, columns=all_items)
    
    # Menambahkan kolom ID Transaksi dan Tahun di awal
    one_hot_df.insert(0, 'ID Transaksi', df['ID Transaksi'])
    one_hot_df.insert(1, 'Tahun', df['Tahun'])
    
    # Menyimpan hasil one-hot encoding ke Excel
    with pd.ExcelWriter(one_hot_path, engine='xlsxwriter') as writer:
        one_hot_df.to_excel(writer, index=False, sheet_name='One-Hot')
        
        # Mengatur lebar kolom dan tinggi baris
        worksheet = writer.sheets['One-Hot']
        for i, col in enumerate(one_hot_df.columns):
            max_len = max(one_hot_df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)
        # Mengatur tinggi default baris menjadi 20
        worksheet.set_default_row(20)
    
    # Mengembalikan dictionary berisi path file-file output
    return {
        'processed_excel': processed_excel_path,
        'json_file': json_path,
        'one_hot_file': one_hot_path
    }

# Bagian main untuk eksekusi script
if __name__ == "__main__":
    # Mendefinisikan file input dan folder output
    input_file = 'transaksi_kosmetik.xlsx'
    output_folder = 'data_temp'
    
    try:
        # Memproses file dan mendapatkan path output
        result_paths = process_excel(input_file, output_folder)
        
        # Menampilkan informasi output
        print("Proses berhasil! File disimpan di:")
        print(f"- File Excel diproses: {result_paths['processed_excel']}")
        print(f"- File JSON basket: {result_paths['json_file']}")
        print(f"- One-hot encoding: {result_paths['one_hot_file']}")
        
    except Exception as e:
        # Menangani error dan memberikan petunjuk troubleshooting
        print(f"Error: {str(e)}")
        print("Pastikan:")
        print("- File input ada di direktori yang benar")
        print("- Tidak ada file output yang sedang terbuka")
        print("- Ada cukup space di storage")