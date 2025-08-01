# Import library yang diperlukan
import pandas as pd
import time
from collections import defaultdict
from itertools import combinations
import os
import pydot
import numpy as np

class FPTreeNode:
    __slots__ = ['nama', 'jumlah', 'induk', 'anak', 'tautan']  # Mengurangi penggunaan memori
    
    def __init__(self, nama, jumlah, induk):
        self.nama = nama  # Nama item
        self.jumlah = jumlah  # Jumlah dukungan
        self.induk = induk  # Node induk
        self.anak = {}  # Kamus node anak
        self.tautan = None  # Tautan ke node dengan nama item yang sama

def visualisasi_fp_tree(akar, tabel_header, inisial_item, path_keluaran):
    """Visualisasi FP-Tree yang dioptimalkan"""
    graph = pydot.Dot(graph_type='digraph', rankdir='TB', simplify=True)
    
    # Pendekatan iteratif untuk menghindari overflow stack
    tumpukan = [(akar, None)]
    while tumpukan:
        node, id_induk = tumpukan.pop()
        
        # Buat label node
        label_node = "Akar" if node.nama is None else f"{inisial_item.get(node.nama, node.nama)}\n({node.jumlah})"
        id_node = str(id(node))
        warna = 'lightblue' if node.nama is None else 'lightgreen'
        
        graph.add_node(pydot.Node(id_node, label=label_node, style='filled', fillcolor=warna))
        
        if id_induk is not None:
            graph.add_edge(pydot.Edge(id_induk, id_node))
        
        # Proses anak secara terbalik untuk urutan kiri-ke-kanan
        for anak in reversed(list(node.anak.values())):
            tumpukan.append((anak, id_node))
    
    graph.write_jpg(path_keluaran)
    return path_keluaran

def algoritma_fp_growth(file_one_hot, dukungan_min=0.3, kepercayaan_min=0.6, direktori_keluaran='keluaran'):
    """Implementasi FP-Growth yang dioptimalkan dengan istilah Bahasa Indonesia"""
    
    # Buat direktori keluaran
    os.makedirs(direktori_keluaran, exist_ok=True)
    
    # Baca data dengan hanya kolom yang diperlukan
    df = pd.read_excel(file_one_hot, sheet_name='One-Hot', 
                      usecols=lambda col: col not in ['Tahun'])  # Abaikan kolom tidak digunakan
    
    # Hitung kolom item dan total transaksi
    kolom_item = [col for col in df.columns if col not in ['ID Transaksi']]
    transaksi = df[kolom_item].values  # Gunakan array numpy untuk operasi lebih cepat
    total_transaksi = len(transaksi)
    jumlah_dukungan_min = dukungan_min * total_transaksi
    
    # Mulai pengukuran waktu
    waktu_mulai = time.perf_counter()
    
    # Langkah 1: Hitung dukungan menggunakan numpy untuk kecepatan
    jumlah_item = pd.Series(transaksi.sum(axis=0), index=kolom_item)
    
    # Filter item yang sering muncul lebih awal
    masker_sering = jumlah_item >= jumlah_dukungan_min
    item_sering = jumlah_item[masker_sering]
    item_terurut = item_sering.sort_values(ascending=False).items()
    
    # Perbaikan: Pastikan masker_sering sesuai dengan kolom_item
    masker_sering_array = np.zeros(len(kolom_item), dtype=bool)
    for i, item in enumerate(kolom_item):
        masker_sering_array[i] = masker_sering.get(item, False)
    
    # Buat pemetaan inisial item
    inisial_item = {}
    inisial_digunakan = set()
    for item, _ in item_terurut:
        inisial = ''.join(kata[0].upper() for kata in item.split())
        # Tangani duplikat
        if inisial in inisial_digunakan:
            i = 1
            while f"{inisial}{i}" in inisial_digunakan:
                i += 1
            inisial = f"{inisial}{i}"
        inisial_digunakan.add(inisial)
        inisial_item[item] = inisial
    
    # Buat DataFrame lebih efisien
    itemset_1_sering = pd.DataFrame({
        'Item': item_sering.index,
        'Jumlah Transaksi': item_sering.values,
        'Support': item_sering.values / total_transaksi
    })
    itemset_1_sering['Keterangan'] = 'Lolos'
    
    # Buat tabel item dengan operasi vektor
    tabel_item = pd.DataFrame({
        'Item': item_sering.index,
        'Inisial': [inisial_item[item] for item in item_sering.index],
        'Jumlah Transaksi': item_sering.values,
        'Support': item_sering.values / total_transaksi
    }).sort_values('Jumlah Transaksi', ascending=False)
    
    # Langkah 2: Proses transaksi dengan numpy untuk kecepatan
    set_item_sering = set(item_sering.index)
    urutan_item = {item: idx for idx, item in enumerate(item_sering.index)}
    
    transaksi_terproses = []
    for baris in transaksi:
        # Dapatkan item sering dengan boolean indexing numpy
        masker = (baris == 1) & masker_sering_array  # Gunakan array yang sudah diperbaiki
        item_transaksi = np.array(kolom_item)[masker].tolist()
        # Urutkan berdasarkan urutan yang sudah dihitung
        item_transaksi.sort(key=lambda x: (-item_sering[x], x))
        transaksi_terproses.append([inisial_item[item] for item in item_transaksi])
    
    tabel_transaksi = pd.DataFrame({
        'ID Transaksi': df['ID Transaksi'],
        'Item (Transaksi terurut)': [', '.join(item) for item in transaksi_terproses]
    })
    
    # Langkah 3: Bangun FP-Tree dengan struktur data optimal
    tabel_header = defaultdict(list)
    akar = FPTreeNode(None, 0, None)
    
    # Hitung urutan item untuk konstruksi pohon
    urutan_item = {item: idx for idx, (item, _) in enumerate(item_terurut)}
    
    for transaksi in transaksi_terproses:
        # Urutkan transaksi berdasarkan urutan global item
        transaksi.sort(key=lambda x: urutan_item.get(x, float('inf')))
        
        node_sekarang = akar
        for item in transaksi:
            if item in node_sekarang.anak:
                anak = node_sekarang.anak[item]
                anak.jumlah += 1
            else:
                anak = FPTreeNode(item, 1, node_sekarang)
                node_sekarang.anak[item] = anak
                # Perbarui tabel header
                if tabel_header[item]:
                    tabel_header[item][-1].tautan = anak
                tabel_header[item].append(anak)
            node_sekarang = anak
    
    # Visualisasikan FP-Tree
    path_gambar_fp_tree = os.path.join(direktori_keluaran, 'fp_tree.jpg')
    visualisasi_fp_tree(akar, tabel_header, inisial_item, path_gambar_fp_tree)
    
    # Langkah 4: Menambang FP-Tree dengan koleksi basis pola optimal
    langkah_penambangan = []
    itemset_sering = []
    
    def tambang_pohon(tabel_header, awalan, level=0):
        """Penambangan pohon yang dioptimalkan dengan alokasi memori lebih sedikit"""
        # Proses item berdasarkan frekuensi (bawah-ke-atas)
        item = sorted(tabel_header.keys(), key=lambda x: len(tabel_header[x]))
        
        for item in item:
            jumlah_dukungan = sum(node.jumlah for node in tabel_header[item])
            awalan_baru = awalan | {item}
            itemset_sering.append((awalan_baru, jumlah_dukungan))
            
            # Kumpulkan basis pola bersyarat
            pola_bersyarat = []
            for node in tabel_header[item]:
                jalur = []
                induk = node.induk
                while induk.nama is not None:
                    jalur.append(induk.nama)
                    induk = induk.induk
                if jalur:
                    pola_bersyarat.append((jalur, node.jumlah))
            
            langkah_penambangan.append({
                'Tahap': 'Basis Pola Bersyarat',
                'Item': item,
                'Basis Pola': [f"{jalur} (jumlah: {jumlah})" for jalur, jumlah in pola_bersyarat],
                'Level': level
            })
            
            # Bangun FP-Tree bersyarat
            jumlah_bersyarat = defaultdict(int)
            for jalur, jumlah in pola_bersyarat:
                for item_dalam_jalur in jalur:
                    jumlah_bersyarat[item_dalam_jalur] += jumlah
            
            # Filter item yang memenuhi dukungan minimum
            item_bersyarat = {item: jumlah for item, jumlah in jumlah_bersyarat.items() 
                            if jumlah >= jumlah_dukungan_min}
            
            if item_bersyarat:
                # Bangun tabel header bersyarat
                header_bersyarat = defaultdict(list)
                akar_bersyarat = FPTreeNode(None, 0, None)
                
                # Urutkan item berdasarkan frekuensi
                item_bersyarat_terurut = sorted(item_bersyarat.items(), 
                                               key=lambda x: (-x[1], x[0]))
                
                for jalur, jumlah in pola_bersyarat:
                    # Filter dan urutkan item
                    jalur_terfilter = [item for item in jalur if item in item_bersyarat]
                    jalur_terfilter.sort(key=lambda x: (-item_bersyarat[x], x))
                    
                    # Perbarui pohon
                    node_sekarang = akar_bersyarat
                    for item in jalur_terfilter:
                        if item in node_sekarang.anak:
                            anak = node_sekarang.anak[item]
                            anak.jumlah += jumlah
                        else:
                            anak = FPTreeNode(item, jumlah, node_sekarang)
                            node_sekarang.anak[item] = anak
                            if header_bersyarat[item]:
                                header_bersyarat[item][-1].tautan = anak
                            header_bersyarat[item].append(anak)
                        node_sekarang = anak
                
                langkah_penambangan.append({
                    'Tahap': 'FP-Tree Bersyarat',
                    'Item': item,
                    'Item Pohon': list(header_bersyarat.keys()),
                    'Level': level
                })
                
                # Tambang pohon secara rekursif
                tambang_pohon(header_bersyarat, awalan_baru, level + 1)
    
    tambang_pohon(tabel_header, set())
    
    # Konversi langkah penambangan ke DataFrame
    df_langkah_penambangan = pd.DataFrame(langkah_penambangan) if langkah_penambangan else pd.DataFrame()
    
    # Langkah 5: Hasilkan aturan asosiasi dengan optimasi
    aturan = []
    kamus_itemset = {frozenset(itemset): dukungan for itemset, dukungan in itemset_sering}
    
    for itemset, jumlah_dukungan in itemset_sering:
        if len(itemset) < 2:
            continue
            
        dukungan = jumlah_dukungan / total_transaksi
        daftar_itemset = list(itemset)
        
        # Hasilkan semua anteseden mungkin
        for i in range(1, len(daftar_itemset)):
            for anteseden in combinations(daftar_itemset, i):
                set_anteseden = frozenset(anteseden)
                set_konsekuen = frozenset(daftar_itemset) - set_anteseden
                
                # Cari dukungan anteseden
                dukungan_anteseden = kamus_itemset.get(set_anteseden, 0) / total_transaksi
                if dukungan_anteseden == 0:
                    continue
                
                kepercayaan = dukungan / dukungan_anteseden
                if kepercayaan < kepercayaan_min:
                    continue
                
                # Cari dukungan konsekuen
                dukungan_konsekuen = kamus_itemset.get(set_konsekuen, 0) / total_transaksi
                lift = kepercayaan / dukungan_konsekuen if dukungan_konsekuen > 0 else 0
                korelasi = "Positif" if lift > 1 else "Negatif" if lift < 1 else "Netral"
                
                # Konversi inisial kembali ke nama asli
                peta_nama_item = {v: k for k, v in inisial_item.items()}
                item_anteseden = [peta_nama_item[a] for a in anteseden]
                item_konsekuen = [peta_nama_item[k] for k in set_konsekuen]
                
                # Hitung metrik
                presisi = kepercayaan
                recall = presisi
                f1 = 2 * presisi * recall / (presisi + recall) if (presisi + recall) > 0 else 0
                
                aturan.append({
                    'Rule': f"{item_anteseden} => {item_konsekuen}",
                    'Support': round(dukungan, 4),
                    'Support A': round(dukungan_anteseden, 4),
                    'Confidence': round(kepercayaan, 4),
                    'Lift': round(lift, 4),
                    'Korelasi': korelasi,
                    'Akurasi': round((presisi + recall) / 2, 4),
                    'Recall': round(recall, 4),
                    'Presisi': round(presisi, 4),
                    'F1-Score': round(f1, 4)
                })
    
    # Hitung metrik
    waktu_eksekusi = time.perf_counter() - waktu_mulai
    jumlah_aturan = len(aturan)
    rata_lift = sum(rule['Lift'] for rule in aturan) / jumlah_aturan if jumlah_aturan > 0 else 0
    rata_akurasi = sum(rule['Akurasi'] for rule in aturan) / jumlah_aturan if jumlah_aturan > 0 else 0
    
    return {
        'itemset_1_sering': itemset_1_sering,
        'tabel_item': tabel_item,
        'tabel_transaksi': tabel_transaksi,
        'gambar_fp_tree': path_gambar_fp_tree,
        'langkah_penambangan': df_langkah_penambangan,
        'aturan_asosiasi': pd.DataFrame(aturan),
        'metrik': {
            'Waktu Eksekusi (detik)': round(waktu_eksekusi, 2),
            'Jumlah Rule Ditemukan': jumlah_aturan,
            'Rata-rata Lift': round(rata_lift, 4),
            'Rata-rata Akurasi': round(rata_akurasi, 4),
            'Min Support': dukungan_min,
            'Min Confidence': kepercayaan_min,
            'Total Transaksi': total_transaksi
        },
        'inisial_item': inisial_item,
        'file_input': file_one_hot
    }

def simpan_hasil_ke_excel(hasil, file_keluaran):
    """Fungsi penyimpanan Excel dengan 2 sheet: One-Hot dan Metode FP-Growth"""
    os.makedirs(os.path.dirname(file_keluaran), exist_ok=True)
    
    with pd.ExcelWriter(file_keluaran, engine='xlsxwriter') as writer:
        # Sheet 1: Original One-Hot data
        one_hot_df = pd.read_excel(hasil['file_input'], sheet_name='One-Hot')
        one_hot_df.to_excel(writer, sheet_name='One-Hot', index=False)
        
        # Sheet 2: Metode FP-Growth (semua hasil)
        workbook = writer.book
        worksheet = workbook.add_worksheet('Metode FP-Growth')
        writer.sheets['Metode FP-Growth'] = worksheet
        
        # Tulis semua hasil ke sheet Metode FP-Growth
        row = 0
        
        # 1. Itemset Sering 1
        worksheet.write(row, 0, '1. Frequent 1-itemset (filter)')
        row += 1
        hasil['itemset_1_sering'].to_excel(writer, sheet_name='Metode FP-Growth', 
                                         startrow=row, index=False)
        row += len(hasil['itemset_1_sering']) + 3
        
        # 2. Item dan Inisial
        worksheet.write(row, 0, '2. Item dan Inisial (Lolos)')
        row += 1
        hasil['tabel_item'].to_excel(writer, sheet_name='Metode FP-Growth',
                                   startrow=row, index=False)
        row += len(hasil['tabel_item']) + 3
        
        # 3. Transaksi yang Diproses
        worksheet.write(row, 0, '3. Transaksi yang Diproses')
        row += 1
        hasil['tabel_transaksi'].to_excel(writer, sheet_name='Metode FP-Growth',
                                       startrow=row, index=False)
        row += len(hasil['tabel_transaksi']) + 3
        
        # 4. Visualisasi FP-Tree
        worksheet.write(row, 0, '4. Visualisasi FP-Tree')
        row += 1
        worksheet.insert_image(row, 0, hasil['gambar_fp_tree'], {'x_scale': 0.5, 'y_scale': 0.5})
        row += 21  # Sesuaikan tinggi gambar
        
        # 5. Tahapan Mining FP-Tree
        worksheet.write(row, 0, '5. Tahapan Mining FP-Tree')
        row += 1
        if not hasil['langkah_penambangan'].empty:
            hasil['langkah_penambangan'].to_excel(writer, sheet_name='Metode FP-Growth',
                                               startrow=row, index=False)
            row += len(hasil['langkah_penambangan']) + 3
        
        # 6. Association Rules
        worksheet.write(row, 0, '6. Association Rules')
        row += 1
        hasil['aturan_asosiasi'].to_excel(writer, sheet_name='Metode FP-Growth',
                                        startrow=row, index=False)
        row += len(hasil['aturan_asosiasi']) + 3
        
        # 7. Metrics
        worksheet.write(row, 0, '7. Metrics')
        row += 1
        pd.DataFrame.from_dict(hasil['metrik'], orient='index', columns=['Value'])\
            .to_excel(writer, sheet_name='Metode FP-Growth',
                     startrow=row, header=True)
        
        # Atur lebar kolom untuk sheet Metode FP-Growth
        worksheet = writer.sheets['Metode FP-Growth']
        df_combined = pd.concat([
            hasil['itemset_1_sering'],
            hasil['tabel_item'],
            hasil['tabel_transaksi'],
            hasil['langkah_penambangan'] if not hasil['langkah_penambangan'].empty else pd.DataFrame(),
            hasil['aturan_asosiasi'],
            pd.DataFrame.from_dict(hasil['metrik'], orient='index')
        ], axis=0)
        
        for idx, col in enumerate(df_combined.columns):
            max_len = max(
                df_combined[col].astype(str).str.len().max(),
                len(str(col))
            ) + 2 if not df_combined.empty else len(str(col)) + 2
            worksheet.set_column(idx, idx, max_len)

if __name__ == "__main__":
    # Konfigurasi
    file_input = 'data_temp/tahap3_one_hot_transaksi.xlsx'
    file_keluaran = 'data_temp/tahap5_hasil_perhitungan.xlsx'
    direktori_keluaran = 'data_temp/fp_growth_output'
    dukungan_min = 0.3
    kepercayaan_min = 0.6
    
    # Jalankan dengan pengukuran waktu
    print("Menjalankan algoritma FP-Growth yang dioptimalkan...")
    waktu_mulai = time.perf_counter()
    hasil = algoritma_fp_growth(file_input, dukungan_min, kepercayaan_min, direktori_keluaran)
    durasi = time.perf_counter() - waktu_mulai
    
    # Simpan hasil
    simpan_hasil_ke_excel(hasil, file_keluaran)
    
    # Cetak ringkasan
    print(f"Hasil disimpan ke: {file_keluaran}")
    
    # Print summary metrics
    print("\nRingkasan Hasil:")
    for kunci, nilai in hasil['metrik'].items():
        print(f"{kunci}: {nilai}")
        
    print(f"\nGambar FP-Tree disimpan di: {hasil['gambar_fp_tree']}")