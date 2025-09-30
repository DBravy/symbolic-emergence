# Multi-Phase Training System

## Overview

The web interface now supports a multi-phase training pipeline where you can add intermediate phases between Phase 0 (Communication Protocol) and the final ARC Solving phase.

## Features

### 1. Dynamic Phase Navigation
- All pages now have dynamic navigation that updates based on your configured phases
- The current page is highlighted in the navigation
- Easy navigation between all training phases

### 2. Phase Management UI
Located on the main page (Phase 0 / Communication Protocol):

- **View all phases**: See your complete training pipeline at a glance
- **Add new phases**: Click "➕ Add New Phase" or the "➕" button next to any phase to insert after it
- **Edit phases**: Click the "✏️" button to rename intermediate phases
- **Delete phases**: Click the "🗑️" button to remove intermediate phases
- **Phase types**:
  - `base`: Phase 0 - Communication Protocol (cannot be deleted)
  - `intermediate`: Custom phases you add (can be edited/deleted)
  - `final`: ARC Solving (cannot be deleted)

### 3. Placeholder Pages
Intermediate phases automatically get placeholder pages that:
- Show the phase name and information
- Include the dynamic navigation
- Can be customized later with specific training interfaces

## File Structure

### Backend (`web_app.py`)
- **Phase storage**: Phases are stored in `training_phases.json`
- **API endpoints**:
  - `GET /api/phases` - Get all phases
  - `POST /api/phases` - Add a new phase
  - `PUT /api/phases/<id>` - Update phase name
  - `DELETE /api/phases/<id>` - Delete a phase
- **Routes**:
  - `/` - Phase 0 (Communication Protocol)
  - `/phase/<id>` - Dynamic phase pages
  - `/arc` - ARC Solving page

### Frontend Templates
- `templates/index.html` - Phase 0 page with phase management UI
- `templates/arc.html` - Final ARC solving page
- `templates/phase_intermediate.html` - Placeholder for intermediate phases

### Styling
- `static/style.css` - Added phase management styles and navigation updates

## Usage

### Starting the Server
```bash
python web_app.py
```
Then open http://localhost:5001

### Adding a Phase
1. Go to the main page (Communication Protocol)
2. Scroll to the "Training Phases" section
3. Click "➕ Add New Phase" (adds before ARC) or "➕" next to a specific phase
4. Enter a name for the phase
5. The phase will appear in the navigation

### Editing a Phase
1. Find the phase in the "Training Phases" list
2. Click the "✏️" edit button
3. Enter a new name

### Deleting a Phase
1. Find the intermediate phase in the list
2. Click the "🗑️" delete button
3. Confirm deletion

### Navigating Between Phases
- Click any phase name in the top navigation bar
- The current phase is highlighted

## Default Configuration

By default, the system starts with two phases:
- Phase 0: Communication Protocol (base)
- ARC Solving (final)

## Customizing Intermediate Phases

To add custom functionality to an intermediate phase:

1. Create a new template in `templates/` (e.g., `phase_custom.html`)
2. Modify the `phase_page()` function in `web_app.py` to route specific phase IDs to your custom template
3. Add any necessary API endpoints for your phase's functionality

Example:
```python
@app.route('/phase/<phase_id>')
def phase_page(phase_id):
    phases = load_phases()
    phase = find_phase_by_id(phase_id)
    
    # Custom routing for specific phases
    if phase_id == '1':
        return render_template('phase_custom.html', phase=phase, phases=phases)
    
    # Default placeholder for others
    return render_template('phase_intermediate.html', phase=phase, phases=phases)
```

## Data Persistence

- Phase configuration is saved to `training_phases.json` in the project root
- Changes are persisted immediately when you add, edit, or delete phases
- The file is loaded on server startup

