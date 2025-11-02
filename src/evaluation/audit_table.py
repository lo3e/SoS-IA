import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# === CONFIG ===
load_dotenv()
DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sosia.db"

# === UTILS ===
def connect_db():
    return sqlite3.connect(DB_PATH)

def get_table_names(conn):
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return [r[0] for r in c.fetchall()]

def audit_table(conn, table):
    """Ritorna info dettagliate su una singola tabella"""
    c = conn.cursor()
    info = {"table": table}

    # numero righe
    c.execute(f"SELECT COUNT(*) FROM {table}")
    info["rows"] = c.fetchone()[0]

    # colonne e tipi
    c.execute(f"PRAGMA table_info({table})")
    cols = c.fetchall()
    columns = []
    for col in cols:
        col_name = col[1]
        col_type = col[2]
        pk = bool(col[5])
        # calcola percentuale NULL
        c.execute(f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL")
        nulls = c.fetchone()[0]
        null_pct = (nulls / info["rows"] * 100) if info["rows"] > 0 else 0
        columns.append({
            "column": col_name,
            "type": col_type,
            "pk": pk,
            "null_pct": round(null_pct, 2)
        })
    info["columns"] = columns
    return info

def main():
    print(f"📊 AUDIT DATABASE STRUCTURE\n{'='*60}")
    conn = connect_db()
    tables = get_table_names(conn)
    print(f"📁 Database: {DB_PATH}")
    print(f"📦 {len(tables)} tabelle trovate\n")

    audit_summary = []

    for t in tables:
        t_info = audit_table(conn, t)
        print(f"🧱 Tabella: {t} ({t_info['rows']} righe)")
        for col in t_info["columns"]:
            pk_flag = " (PK)" if col["pk"] else ""
            print(f"   • {col['column']:<20} {col['type']:<10} NULL%={col['null_pct']:<6}{pk_flag}")
        print("-" * 60)
        audit_summary.append({
            "table": t,
            "rows": t_info["rows"],
            "columns": len(t_info["columns"]),
            "avg_null_pct": round(sum(c["null_pct"] for c in t_info["columns"]) / len(t_info["columns"]), 2)
        })

    # riepilogo finale
    df_summary = pd.DataFrame(audit_summary)
    print("\n📈 RIEPILOGO GENERALE")
    print(df_summary.to_string(index=False))

    # salva anche su file CSV per analisi successive
    out_path = Path(__file__).resolve().parent / "db_audit_report.csv"
    df_summary.to_csv(out_path, index=False)
    print(f"\n📝 Report salvato in: {out_path}")

    conn.close()
    print("\n✅ Audit completato.")

if __name__ == "__main__":
    main()
