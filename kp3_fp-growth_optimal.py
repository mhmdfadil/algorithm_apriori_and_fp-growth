# Import library yang diperlukan
import pandas as pd
import time
from collections import defaultdict
from itertools import combinations
import os
import pydot
import numpy as np

class FPTreeNode:
    __slots__ = ['nama', 'jumlah', 'induk', 'anak', 'tautan']
    
    def __init__(self, nama, jumlah, induk):
        self.nama = nama
        self.jumlah = jumlah
        self.induk = induk
        self.anak = {}
        self.tautan = None

def visualisasi_fp_tree(akar, tabel_header, inisial_item, path_keluaran):
    """Visualisasi FP-Tree yang dioptimalkan"""
    graph = pydot.Dot(graph_type='digraph', rankdir='TB', simplify=True)
    
    # Pendekatan iteratif untuk menghindari overflow stack
    tumpukan = [(akar, None)]
    node_count = 0
    node_map = {}
    
    while tumpukan:
        node, id_induk = tumpukan.pop()
        
        # Buat label node
        if node.nama is None:
            label_node = "Akar"
            warna = 'lightblue'
        else:
            label_node = f"{inisial_item.get(node.nama, node.nama)}\n({node.jumlah})"
            warna = 'lightgreen'
        
        # Gunakan ID yang lebih sederhana
        node_id = f"node{node_count}"
        node_count += 1
        node_map[node] = node_id
        
        graph.add_node(pydot.Node(node_id, label=label_node, style='filled', fillcolor=warna))
        
        if id_induk is not None:
            graph.add_edge(pydot.Edge(id_induk, node_id))
        
        # Proses anak secara terbalik untuk urutan kiri-ke-kanan
        for anak in reversed(list(node.anak.values())):
            tumpukan.append((anak, node_id))
    
    graph.write_jpg(path_keluaran)
    return path_keluaran

def algoritma_fp_growth(file_one_hot, dukungan_min=0.3, kepercayaan_min=0.6, direktori_keluaran='keluaran'):
    """Implementasi FP-Growth yang dioptimalkan"""
    
    os.makedirs(direktori_keluaran, exist_ok=True)
    
    # Baca data dengan optimasi memori
    df = pd.read_excel(file_one_hot, sheet_name='One-Hot')
    kolom_item = [col for col in df.columns if col not in ['ID Transaksi', 'Tahun']]
    transaksi_data = df[kolom_item].values
    total_transaksi = len(transaksi_data)
    jumlah_dukungan_min = int(dukungan_min * total_transaksi)
    
    
    
    # Langkah 1: Hitung dukungan dengan numpy
    jumlah_item = np.sum(transaksi_data, axis=0)
    item_counts_dict = dict(zip(kolom_item, jumlah_item))
    
    # Filter item yang memenuhi dukungan minimum
    mask_sering = jumlah_item >= jumlah_dukungan_min
    item_sering = [item for item, mask in zip(kolom_item, mask_sering) if mask]
    jumlah_item_sering = [count for count, mask in zip(jumlah_item, mask_sering) if mask]
    
    # Urutkan item berdasarkan frekuensi (descending)
    sorted_indices = np.argsort(jumlah_item_sering)[::-1]
    item_sering_terurut = [item_sering[i] for i in sorted_indices]
    jumlah_item_sering_terurut = [jumlah_item_sering[i] for i in sorted_indices]
    
    # Buat pemetaan inisial
    inisial_item = {}
    inisial_digunakan = set()
    for item in item_sering_terurut:
        words = item.split()
        initial = ''.join([word[0].upper() for word in words])
        
        if initial in inisial_digunakan:
            i = 1
            while f"{initial}{i}" in inisial_digunakan:
                i += 1
            initial = f"{initial}{i}"
        
        inisial_digunakan.add(initial)
        inisial_item[item] = initial
    
    # Buat tabel itemset 1-sering
    itemset_1_sering = pd.DataFrame({
        'Item': item_sering_terurut,
        'Jumlah Transaksi': jumlah_item_sering_terurut,
        'Support': [count/total_transaksi for count in jumlah_item_sering_terurut],
        'Keterangan': 'Lolos'
    })
    
    # Buat tabel item dengan inisial
    tabel_item = pd.DataFrame({
        'Item': item_sering_terurut,
        'Inisial': [inisial_item[item] for item in item_sering_terurut],
        'Jumlah Transaksi': jumlah_item_sering_terurut,
        'Support': [count/total_transaksi for count in jumlah_item_sering_terurut]
    })
    
    # Langkah 2: Proses transaksi
    # Buat mapping untuk akses cepat
    item_to_idx = {item: idx for idx, item in enumerate(kolom_item)}
    sering_mask = np.array([item in item_sering_terurut for item in kolom_item])
    
    transaksi_terproses = []
    for i in range(total_transaksi):
        transaksi_baris = transaksi_data[i]
        item_transaksi = []
        
        for j, item in enumerate(kolom_item):
            if sering_mask[j] and transaksi_baris[j] == 1:
                item_transaksi.append(item)
        
        # Urutkan berdasarkan frekuensi (descending)
        item_transaksi.sort(key=lambda x: -item_counts_dict[x])
        transaksi_terproses.append([inisial_item[item] for item in item_transaksi])
    
    tabel_transaksi = pd.DataFrame({
        'ID Transaksi': df['ID Transaksi'],
        'Item (Transaksi terurut)': [', '.join(items) for items in transaksi_terproses]
    })
    
    # Langkah 3: Bangun FP-Tree
    tabel_header = defaultdict(list)
    akar = FPTreeNode(None, 0, None)
    
    # Buat urutan global untuk sorting
    urutan_global = {item: idx for idx, item in enumerate(item_sering_terurut)}
    
    for transaksi in transaksi_terproses:
        # Urutkan transaksi berdasarkan urutan global
        transaksi.sort(key=lambda x: urutan_global.get(x, float('inf')))
        
        node_sekarang = akar
        for item in transaksi:
            if item in node_sekarang.anak:
                child = node_sekarang.anak[item]
                child.jumlah += 1
            else:
                child = FPTreeNode(item, 1, node_sekarang)
                node_sekarang.anak[item] = child
                
                # Update header table
                if tabel_header[item]:
                    last_node = tabel_header[item][-1]
                    last_node.tautan = child
                tabel_header[item].append(child)
            
            node_sekarang = node_sekarang.anak[item]
    
    # Visualisasi FP-Tree
    path_gambar_fp_tree = os.path.join(direktori_keluaran, 'fp_tree.jpg')
    visualisasi_fp_tree(akar, tabel_header, inisial_item, path_gambar_fp_tree)
    
    waktu_mulai_mining = time.perf_counter()
    
    # Langkah 4: Mining FP-Tree
    langkah_penambangan = []
    semua_itemset_sering = []
    
    def mine_tree(current_header, prefix, level=0):
        items = list(current_header.keys())
        
        for item in items:
            support_count = sum(node.jumlah for node in current_header[item])
            new_prefix = prefix | {item}
            semua_itemset_sering.append((frozenset(new_prefix), support_count))
            
            # Kumpulkan conditional pattern base
            conditional_patterns = []
            for node in current_header[item]:
                path = []
                count = node.jumlah
                parent = node.induk
                
                while parent and parent.nama is not None:
                    path.append(parent.nama)
                    parent = parent.induk
                
                if path:
                    conditional_patterns.append((path[::-1], count))
            
            # Simpan langkah mining
            pattern_strs = [f"{path} (count: {count})" for path, count in conditional_patterns]
            langkah_penambangan.append({
                'Tahap': 'Conditional Pattern Base',
                'Item': item,
                'Pattern Base': pattern_strs,
                'Level': level
            })
            
            # Bangun conditional FP-Tree
            item_counts = defaultdict(int)
            for path, count in conditional_patterns:
                for path_item in path:
                    item_counts[path_item] += count
            
            # Filter items yang memenuhi minimum support
            frequent_items = {item: count for item, count in item_counts.items() 
                            if count >= jumlah_dukungan_min}
            
            if frequent_items:
                # Build conditional header table
                cond_header = defaultdict(list)
                cond_root = FPTreeNode(None, 0, None)
                
                # Sort items by frequency
                sorted_frequent = sorted(frequent_items.items(), key=lambda x: (-x[1], x[0]))
                
                for path, count in conditional_patterns:
                    # Filter and sort path items
                    filtered_path = [item for item in path if item in frequent_items]
                    filtered_path.sort(key=lambda x: (-frequent_items[x], x))
                    
                    # Update tree
                    current_node = cond_root
                    for path_item in filtered_path:
                        if path_item in current_node.anak:
                            child_node = current_node.anak[path_item]
                            child_node.jumlah += count
                        else:
                            child_node = FPTreeNode(path_item, count, current_node)
                            current_node.anak[path_item] = child_node
                            
                            if cond_header[path_item]:
                                cond_header[path_item][-1].tautan = child_node
                            cond_header[path_item].append(child_node)
                        
                        current_node = child_node
                
                # Simpan informasi conditional tree
                langkah_penambangan.append({
                    'Tahap': 'Conditional FP-Tree',
                    'Item': item,
                    'Tree Items': list(cond_header.keys()),
                    'Level': level
                })
                
                # Rekursif mining
                mine_tree(cond_header, new_prefix, level + 1)
    
    # Mulai mining dari root
    mine_tree(tabel_header, set())
    
    # Konversi ke DataFrame
    df_langkah_penambangan = pd.DataFrame(langkah_penambangan)
    
    # Langkah 5: Generate association rules
    aturan = []
    itemset_dict = {itemset: support for itemset, support in semua_itemset_sering}
    
    # Generate rules hanya dari itemset dengan 2+ items
    for itemset, support_count in semua_itemset_sering:
        if len(itemset) < 2:
            continue
            
        itemset_list = list(itemset)
        support = support_count / total_transaksi
        
        # Generate all possible antecedents
        for i in range(1, len(itemset_list)):
            for antecedent in combinations(itemset_list, i):
                antecedent_set = frozenset(antecedent)
                consequent_set = itemset - antecedent_set
                
                # Find antecedent support
                ant_support_count = 0
                for itemset2, count in semua_itemset_sering:
                    if antecedent_set.issubset(itemset2):
                        ant_support_count = count
                        break
                
                if ant_support_count == 0:
                    continue
                    
                ant_support = ant_support_count / total_transaksi
                confidence = support / ant_support
                
                if confidence >= kepercayaan_min:
                    # Find consequent support
                    cons_support_count = 0
                    for itemset2, count in semua_itemset_sering:
                        if consequent_set.issubset(itemset2):
                            cons_support_count = count
                            break
                    
                    cons_support = cons_support_count / total_transaksi if cons_support_count > 0 else 0
                    lift = confidence / cons_support if cons_support > 0 else 0
                    
                    korelasi = "Positif" if lift > 1 else "Negatif" if lift < 1 else "Netral"
                    
                    # Convert back to original item names
                    reverse_mapping = {v: k for k, v in inisial_item.items()}
                    ant_items = [reverse_mapping[item] for item in antecedent]
                    cons_items = [reverse_mapping[item] for item in consequent_set]
                    
                    # Calculate metrics
                    precision = confidence
                    recall = precision
                    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    aturan.append({
                        'Rule': f"{ant_items} => {cons_items}",
                        'Support': round(support, 4),
                        'Support A': round(ant_support, 4),
                        'Confidence': round(confidence, 4),
                        'Lift': round(lift, 4),
                        'Korelasi': korelasi,
                        'Akurasi': round((precision + recall) / 2, 4),
                        'Recall': round(recall, 4),
                        'Presisi': round(precision, 4),
                        'F1-Score': round(f1_score, 4)
                    })
    
    # Hitung metrics
    waktu_eksekusi = time.perf_counter() - waktu_mulai_mining
    jumlah_aturan = len(aturan)
    
    metrics = {
        'Waktu Eksekusi (detik)': round(waktu_eksekusi, 2),
        'Jumlah Rule Ditemukan': jumlah_aturan,
        'Rata-rata Lift': round(sum(rule['Lift'] for rule in aturan) / jumlah_aturan if jumlah_aturan > 0 else 0, 4),
        'Rata-rata Akurasi': round(sum(rule['Akurasi'] for rule in aturan) / jumlah_aturan if jumlah_aturan > 0 else 0, 4),
        'Min Support': dukungan_min,
        'Min Confidence': kepercayaan_min,
        'Total Transaksi': total_transaksi
    }
    
    return {
        'itemset_1_sering': itemset_1_sering,
        'tabel_item': tabel_item,
        'tabel_transaksi': tabel_transaksi,
        'gambar_fp_tree': path_gambar_fp_tree,
        'langkah_penambangan': df_langkah_penambangan,
        'aturan_asosiasi': pd.DataFrame(aturan),
        'metrik': metrics,
        'inisial_item': inisial_item,
        'file_input': file_one_hot
    }

def simpan_hasil_ke_excel(hasil, file_keluaran):
    """Simpan hasil ke Excel dengan format yang rapi"""
    os.makedirs(os.path.dirname(file_keluaran), exist_ok=True)
    
    with pd.ExcelWriter(file_keluaran, engine='xlsxwriter') as writer:
        # Sheet 1: One-Hot
        df_original = pd.read_excel(hasil['file_input'], sheet_name='One-Hot')
        df_original.to_excel(writer, sheet_name='One-Hot', index=False)
        
        # Sheet 2: Metode FP-Growth
        workbook = writer.book
        worksheet = workbook.add_worksheet('Metode FP-Growth')
        
        row = 0
        # 1. Frequent 1-itemset
        worksheet.write(row, 0, '1. Frequent 1-Itemset (Filter: Lolos)')
        row += 1
        hasil['itemset_1_sering'].to_excel(writer, sheet_name='Metode FP-Growth', 
                                         startrow=row, index=False)
        row += len(hasil['itemset_1_sering']) + 3
        
        # 2. Tabel Item
        worksheet.write(row, 0, '2. Tabel Item dengan Inisial')
        row += 1
        hasil['tabel_item'].to_excel(writer, sheet_name='Metode FP-Growth',
                                   startrow=row, index=False)
        row += len(hasil['tabel_item']) + 3
        
        # 3. Transaksi Terproses
        worksheet.write(row, 0, '3. Transaksi yang Telah Diproses')
        row += 1
        hasil['tabel_transaksi'].to_excel(writer, sheet_name='Metode FP-Growth',
                                       startrow=row, index=False)
        row += len(hasil['tabel_transaksi']) + 3
        
        # 4. Langkah-langkah Mining
        if not hasil['langkah_penambangan'].empty:
            worksheet.write(row, 0, '4. Langkah-langkah Mining FP-Tree')
            row += 1
            hasil['langkah_penambangan'].to_excel(writer, sheet_name='Metode FP-Growth',
                                                 startrow=row, index=False)
            row += len(hasil['langkah_penambangan']) + 3
        
        # 5. Aturan Asosiasi
        worksheet.write(row, 0, '5. Aturan Asosiasi')
        row += 1
        hasil['aturan_asosiasi'].to_excel(writer, sheet_name='Metode FP-Growth',
                                        startrow=row, index=False)
        row += len(hasil['aturan_asosiasi']) + 3
        
        # 6. Metrics
        worksheet.write(row, 0, '6. Metrics')
        row += 1
        df_metrics = pd.DataFrame.from_dict(hasil['metrik'], orient='index', columns=['Value'])
        df_metrics.to_excel(writer, sheet_name='Metode FP-Growth',
                          startrow=row, header=True)
        
        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df_original.columns if sheet_name == 'One-Hot' else 
                                     pd.concat([v for k, v in hasil.items() 
                                              if isinstance(v, pd.DataFrame)], axis=0).columns):
                max_len = max(
                    df_original[col].astype(str).str.len().max() if sheet_name == 'One-Hot' else
                    max([v[col].astype(str).str.len().max() for k, v in hasil.items() 
                        if isinstance(v, pd.DataFrame) and col in v.columns]),
                    len(str(col))
                ) + 2
                worksheet.set_column(idx, idx, max_len)

if __name__ == "__main__":
    # Konfigurasi
    file_input = 'data_temp/tahap3_one_hot_transaksi.xlsx'
    file_keluaran = 'data_temp/tahap5_hasil_perhitungan.xlsx'
    direktori_keluaran = 'data_temp/fp_growth_output'
    dukungan_min = 0.3
    kepercayaan_min = 0.6
    
    print("Menjalankan algoritma FP-Growth...")
    hasil = algoritma_fp_growth(file_input, dukungan_min, kepercayaan_min, direktori_keluaran)
    
    simpan_hasil_ke_excel(hasil, file_keluaran)
    
    print(f"Hasil disimpan ke: {file_keluaran}")
    print("\nRingkasan Hasil:")
    for kunci, nilai in hasil['metrik'].items():
        print(f"{kunci}: {nilai}")
    print(f"\nGambar FP-Tree disimpan di: {hasil['gambar_fp_tree']}")