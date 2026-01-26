import sqlite3
import os
import pickle

DB_PATH = 'vision_memory.db'

def get_known_faces():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT person_id, encoding FROM face_encodings")
    data = [(pid, pickle.loads(blob)) for pid, blob in c.fetchall()]
    conn.close()
    return data

def create_new_person(encoding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO persons (name) VALUES (?)", ("Unknown",))
    new_id = c.lastrowid
    c.execute("INSERT INTO face_encodings (person_id, encoding) VALUES (?, ?)", 
              (new_id, pickle.dumps(encoding)))
    conn.commit()
    conn.close()
    return new_id

def get_people_info():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, thumbnail_path FROM persons ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return data

def get_person_name(person_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM persons WHERE id=?", (person_id,))
    res = c.fetchone()
    conn.close()
    return res[0] if res else "Unknown"

def update_thumbnail_path(person_id, path):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE persons SET thumbnail_path = ? WHERE id = ?", (path, person_id))
    conn.commit()
    conn.close()
    
def update_name(person_id, new_name):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE persons SET name=? WHERE id=?", (new_name, person_id))
    conn.commit()
    conn.close()

def delete_person(person_id, thumbnail_path):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
    conn.execute("DELETE FROM face_encodings WHERE person_id=?", (person_id,))
    conn.commit()
    if thumbnail_path and os.path.exists(thumbnail_path):
        os.remove(thumbnail_path)
        
def get_people_count():
    conn = sqlite3.connect(DB_PATH)
    count  = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    conn.close()
    return count

def merge_identities(target_id, source_id):
    conn = sqlite3.connect(DB_PATH)
    
    conn.execute("UPDATE face_encodings SET person_id=? WHERE person_id=?", (target_id, source_id))
    
    res = conn.execute("SELECT thumbnail_path FROM persons WHERE id=?", (source_id,)).fetchone()
    if res and res[0] and os.path.exists(res[0]):
        os.remove(res[0])
        
    conn.execute("DELETE FROM persons WHERE id=?", (source_id,))
    conn.commit()
    conn.close()
    
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS persons
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, thumbnail_path TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS face_encodings
                 (person_id INTEGER, encoding BLOB, FOREIGN KEY(person_id) REFERENCES persons(id))''')
                  
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
    if not os.path.exists('captures'):
        os.makedirs('captures')