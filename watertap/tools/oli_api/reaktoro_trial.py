import reaktoro as rkt
import pyomo.environ as pyo

from reaktoro_softening import construct_state, setup_equilibrium_problem, solve_equilibrium_problem

########################################################################################
# create initial state with no solids formed
print("Aqueous Molality: ")
aq_molalities = {}
initial_state = construct_state(db, temp, pres, inflows)
initial_specs = setup_equilibrium_problem(initial_state, ph=True)
solve_equilibrium_problem(initial_state, initial_specs, set_ph=7.3)
aq_props = rkt.AqueousProps(initial_state.props())
for species in initial_state.system().species():
    aq_molalities[species.name()] = aq_props.speciesMolality(species.name()).val()
    print(f"- {species.name():4}\t{aq_props.speciesMolality(species.name())} mol/kg")
print()
########################################################################################
# create identical system with solids formed
print("Solids Formed:")
state_with_solids = construct_state(db, temp, pres, inflows, allow_solids=True)
specs_with_solids = setup_equilibrium_problem(state_with_solids, ph=True)
solve_equilibrium_problem(state_with_solids, specs_with_solids, set_ph=7.3)
aq_props_with_solids = rkt.AqueousProps(state_with_solids.props())
for species, molality in aq_molalities.items():
    molality_with_solids = aq_props_with_solids.speciesMolality(species).val()
    print(f"- {species:4}\t{int((molality - molality_with_solids) / molality * 100):3}% diff")
print()
########################################################################################
def _add_dose(props, dosant):
    dose = props.speciesAmount(dosant) - initial_state.props().speciesAmount(dosant)
    return dose
#def _ph_oh(props, idx):
#    ph = 14 - (-props.speciesActivityLg(idx))
#    return ph
def _alk_removal(props):
    alk_init = rkt.AqueousProps(initial_state.props()).alkalinity()
    alkalinity = rkt.AqueousProps(props).alkalinity()
    return (alk_init - alkalinity) / alk_init
########################################################################################
# create identical state with solids formed and custom constraints for lime softening
print("With Softening:")
state_with_softening = construct_state(db, temp, pres, inflows, allow_solids=True)
specs_with_softening = setup_equilibrium_problem(state_with_softening)
setup_lime_softening(state_with_softening, specs_with_softening)
solve_equilibrium_problem(
    state_with_softening, specs_with_softening,
    custom_constraints=[("lime", 2.), ("alk_rem", .7)] #, ("soda", 2.) , ("ph_oh", 9.)],
)
aq_props_with_softening = rkt.AqueousProps(state_with_softening.props())
for species, molality in aq_molalities.items():
    molality_with_softening = aq_props_with_softening.speciesMolality(species).val()
    print(f"- {species:4}\t{int((molality - molality_with_softening) / molality * 100):3}% diff")
print()
########################################################################################
# do not allow to form
#excluded_solids = ["Quartz", "Chalcedony"]
# just for reference
#included_solids = ["Aragonite", "Calcite", "Brucite", "Dolomite", "Anhydrite", "Gypsum", "Magnesite"]
########################################################################################
print("Lime softening result:")
ph = aq_props_with_softening.pH().val()
print(f"- pH: {round(ph, 3)}")
dCa = (state_with_softening.props().speciesAmount("Ca+2") - initial_state.props().speciesAmount("Ca+2")).val()
dOH = (state_with_softening.props().speciesAmount("OH-") - initial_state.props().speciesAmount("OH-")).val()
print(f"- Ca+2 added: {round(dCa, 5)} mol")
print(f"- OH- added: {round(dOH, 5)} mol")
alk = aq_props_with_softening.alkalinity().val()
alk_init = aq_props.alkalinity().val()
alk_pct = ((alk_init - alk) / alk_init) * 100
print(alk, alk_init)
print(f"- Alk quench: {round(alk_pct, 0)}% ({round(alk_init - alk, 5)} eq/L)")
