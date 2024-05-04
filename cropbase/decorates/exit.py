# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.
import atexit


@atexit.register
def exit_handler() -> None:
    print("We're exiting now!")


def main() -> None:
    for i in range(10):
        print(2**i)

import atexit
import sqlite3

cxn = sqlite3.connect("db.sqlite3")


def init_db():
    cxn.execute("CREATE TABLE IF NOT EXISTS memes (id INTEGER PRIMARY KEY, meme TEXT)")
    print("Database initialised!")


@atexit.register
def exit_handler():
    cxn.commit()
    cxn.close()
    print("Closed database!")



if __name__ == "__main__":
    init_db()
    1 / 0

    main()
    atexit.unregister(exit_handler)
    1 / 0