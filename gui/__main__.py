from gui.main import main

# Guarded: Step 2's spawned worker processes re-import this module (as
# "__mp_main__"), and must not start another GUI.
if __name__ == "__main__":
    main()
